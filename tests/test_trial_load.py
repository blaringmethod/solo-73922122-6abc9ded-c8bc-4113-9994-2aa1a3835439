"""Trial-load API tests.

Covers: success row counts, immediate constraints (unique / NOT NULL /
NOT DEFERRABLE FK), deferred foreign keys checked at COMMIT, no partial
batch on failure, request validation, job-state gating, package/download
immutability, and concurrent trial loads of the same job.
"""
from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.jobs import STORE, run_job
from app.main import app
from app.models import GenerateRequest, JobStatus, TrialRow
from app.trial import run_trial_load

client = TestClient(app)


def _load(name):
    raw = json.load(open(f"/workspace/examples/{name}"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _run(req_dict):
    req = GenerateRequest(**req_dict)
    jid = STORE.create()
    run_job(jid, req)
    return jid, STORE.get(jid)


@pytest.fixture(scope="module")
def cyclic_job(source_dsn):
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    return jid


def _trial(jid, rows):
    return client.post(f"/jobs/{jid}/trial-load", json={"rows": rows})


def _zip_entries(data: bytes) -> dict[str, str]:
    """Zip bytes -> {name: text}; zip timestamps make raw-byte compares flaky."""
    with zipfile.ZipFile(BytesIO(data)) as zf:
        return {n: zf.read(n).decode() for n in zf.namelist()}


def _base_counts(jid) -> dict:
    """An empty batch is legal: it reports the package's own row counts."""
    r = _trial(jid, [])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    return body["row_counts"]


# A fully valid 4-row batch: every FK target is created earlier in the same
# ordered batch, nullable columns are explicitly null.
GOOD_ROWS = [
    {"table": "dept", "values": {"dept_id": 90001, "dept_code": 7, "parent_dept": None, "mgr_emp": None}},
    {"table": "employee", "values": {"emp_id": 90002, "dept_id": 90001, "mentor_id": None, "level": 3}},
    {"table": "project", "values": {"project_id": 90003, "dept_id": 90001, "dept_code": 7, "budget": None}},
    {"table": "assignment", "values": {"emp_id": 90002, "project_id": 90003, "role_code": None}},
]


def test_trial_load_success_reports_counts(cyclic_job):
    base = _base_counts(cyclic_job)
    r = _trial(cyclic_job, GOOD_ROWS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["failure"] is None
    for t in ("dept", "employee", "project", "assignment"):
        assert body["row_counts"][t] == base[t] + 1


def test_trial_load_example_file(cyclic_job):
    base = _base_counts(cyclic_job)
    r = _trial(cyclic_job, _load("trial_load_rows.json")["rows"])
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert r.json()["row_counts"]["dept"] == base["dept"] + 1


def test_immediate_unique_violation_reports_row(cyclic_job):
    rows = [
        {"table": "dept", "values": {"dept_id": 91001, "dept_code": 8, "parent_dept": None, "mgr_emp": None}},
        {"table": "dept", "values": {"dept_id": 91001, "dept_code": 9, "parent_dept": None, "mgr_emp": None}},
    ]
    body = _trial(cyclic_job, rows).json()
    assert body["ok"] is False
    f = body["failure"]
    assert f["phase"] == "row"
    assert f["row_index"] == 1
    assert f["table"] == "dept"
    assert f["sqlstate"] == "23505"
    assert f["constraint"] == "dept_pk"
    assert f["message"]


def test_immediate_not_null_violation(cyclic_job):
    rows = [
        {"table": "employee", "values": {"emp_id": 91002, "dept_id": 91001, "mentor_id": None, "level": None}},
    ]
    body = _trial(cyclic_job, rows).json()
    assert body["ok"] is False
    f = body["failure"]
    assert f["phase"] == "row" and f["row_index"] == 0
    assert f["sqlstate"] == "23502"
    assert "level" in f["message"]


def test_immediate_fk_violation(cyclic_job):
    rows = [
        {"table": "employee", "values": {"emp_id": 91003, "dept_id": 424242, "mentor_id": None, "level": 2}},
    ]
    body = _trial(cyclic_job, rows).json()
    assert body["ok"] is False
    f = body["failure"]
    assert f["phase"] == "row" and f["row_index"] == 0
    assert f["sqlstate"] == "23503"
    assert f["constraint"] == "employee_dept_fk"


def test_failure_leaves_no_partial_batch(cyclic_job):
    base = _base_counts(cyclic_job)
    rows = [
        {"table": "dept", "values": {"dept_id": 92001, "dept_code": 8, "parent_dept": None, "mgr_emp": None}},
        {"table": "dept", "values": {"dept_id": 92002, "dept_code": 8, "parent_dept": None, "mgr_emp": None}},
        # row 3 violates the NOT DEFERRABLE FK -> whole batch rolls back
        {"table": "employee", "values": {"emp_id": 92003, "dept_id": 424242, "mentor_id": None, "level": 2}},
    ]
    body = _trial(cyclic_job, rows).json()
    assert body["ok"] is False
    assert body["failure"]["row_index"] == 2
    # The two good rows before the failure left nothing behind...
    assert _base_counts(cyclic_job) == base
    # ...and can be re-tried cleanly (a leftover batch would PK-conflict).
    r2 = _trial(cyclic_job, rows[:2])
    assert r2.json()["ok"] is True
    assert r2.json()["row_counts"]["dept"] == base["dept"] + 2


# --- deferred foreign keys (checked only at COMMIT) -------------------------

DEFERRED_DDL = """
CREATE SCHEMA IF NOT EXISTS deferred_demo;
DROP TABLE IF EXISTS deferred_demo.child;
DROP TABLE IF EXISTS deferred_demo.parent;
CREATE TABLE deferred_demo.parent (
    pid integer NOT NULL,
    CONSTRAINT parent_pk PRIMARY KEY (pid)
);
CREATE TABLE deferred_demo.child (
    cid integer NOT NULL,
    pid integer,
    CONSTRAINT child_pk PRIMARY KEY (cid),
    CONSTRAINT child_parent_fk FOREIGN KEY (pid)
        REFERENCES deferred_demo.parent (pid) MATCH SIMPLE
        ON DELETE NO ACTION ON UPDATE NO ACTION DEFERRABLE INITIALLY DEFERRED
);
"""


@pytest.fixture(scope="module")
def deferred_job(source_dsn):
    """A real end-to-end job over a schema with INITIALLY DEFERRED FK."""
    # Structures are created as superuser; the API itself keeps using the
    # read-only app_ro role for introspection.
    info = psycopg.conninfo.conninfo_to_dict(source_dsn)
    info["user"] = "postgres"
    info.pop("password", None)
    admin_dsn = psycopg.conninfo.make_conninfo(**info)
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.cursor().execute(DEFERRED_DDL)
        conn.cursor().execute("GRANT USAGE ON SCHEMA deferred_demo TO app_ro")
    req = {
        "schema_name": "deferred_demo",
        "tables": [
            {"table": "parent", "rows": 2, "ranges": {"pid": {"low": 1, "high": 9}}},
            {
                "table": "child",
                "rows": 2,
                "ranges": {"cid": {"low": 1, "high": 9}, "pid": {"low": 1, "high": 9}},
            },
        ],
    }
    jid, job = _run(req)
    assert job.status == JobStatus.SUCCEEDED, job.conflict
    return jid


def test_deferred_fk_fails_at_commit_phase(deferred_job):
    body = _trial(deferred_job, [
        {"table": "child", "values": {"cid": 100, "pid": 777}},
    ]).json()
    assert body["ok"] is False
    f = body["failure"]
    assert f["phase"] == "commit"
    assert f["row_index"] is None
    assert f["sqlstate"] == "23503"
    assert f["constraint"] == "child_parent_fk"


def test_deferred_fk_parent_then_child_same_batch_succeeds(deferred_job):
    base = _base_counts(deferred_job)
    body = _trial(deferred_job, [
        {"table": "parent", "values": {"pid": 500}},
        {"table": "child", "values": {"cid": 100, "pid": 500}},
    ]).json()
    assert body["ok"] is True
    assert body["row_counts"]["parent"] == base["parent"] + 1
    assert body["row_counts"]["child"] == base["child"] + 1


# --- request validation and job-state gating --------------------------------


def test_validation_rejects_bad_requests(cyclic_job):
    good = {"dept_id": 1, "dept_code": 1, "parent_dept": None, "mgr_emp": None}
    # table not in the job's selected set
    assert _trial(cyclic_job, [{"table": "nope", "values": {}}]).status_code == 422
    # missing column
    bad = dict(good)
    del bad["mgr_emp"]
    assert _trial(cyclic_job, [{"table": "dept", "values": bad}]).status_code == 422
    # unknown extra column
    bad = dict(good, ghost=1)
    assert _trial(cyclic_job, [{"table": "dept", "values": bad}]).status_code == 422
    # more than 20 rows
    assert _trial(cyclic_job, [{"table": "dept", "values": good}] * 21).status_code == 422
    # JSON boolean is not an integer value
    bad = dict(good, dept_id=True)
    assert _trial(cyclic_job, [{"table": "dept", "values": bad}]).status_code == 422


def test_trial_load_requires_succeeded_job(source_dsn):
    jid, job = _run(_load("request_unsat_conflict.json"))
    assert job.status == JobStatus.FAILED
    assert _trial(jid, []).status_code == 409
    assert _trial("0" * 32, []).status_code == 404


# --- isolation: package immutability and concurrency ------------------------


def test_trial_load_does_not_change_package_or_download(cyclic_job):
    jid = cyclic_job
    pkg_before = dict(STORE.package(jid))
    _trial(jid, GOOD_ROWS)
    _trial(jid, [{"table": "dept", "values": {"dept_id": 93001, "dept_code": 1, "parent_dept": None, "mgr_emp": None}}])
    assert STORE.package(jid) == pkg_before
    r = client.get(f"/jobs/{jid}/download")
    assert r.status_code == 200
    assert _zip_entries(r.content) == pkg_before


def test_concurrent_trial_loads_do_not_interfere(cyclic_job):
    jid = cyclic_job
    job = STORE.get(jid)
    package = STORE.package(jid)
    schemas = STORE.schemas(jid)
    pkg_before = dict(package)
    base_dept = 3  # requested dept rows in request_cyclic_four_tables.json

    def one(k: int):
        if k % 2 == 0:
            rows = [TrialRow(table="dept", values={
                "dept_id": 70000 + k, "dept_code": 3, "parent_dept": None, "mgr_emp": None,
            })]
        else:
            # self-conflicting pair: row 1 duplicates row 0's PK
            rows = [
                TrialRow(table="dept", values={
                    "dept_id": 80000, "dept_code": 3, "parent_dept": None, "mgr_emp": None,
                }),
                TrialRow(table="dept", values={
                    "dept_id": 80000, "dept_code": 3, "parent_dept": None, "mgr_emp": None,
                }),
            ]
        return run_trial_load(jid, "public", job.selected_tables, schemas, package, rows)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, range(8)))

    for k, res in enumerate(results):
        if k % 2 == 0:
            assert res.ok, res.failure
            assert res.row_counts["dept"] == base_dept + 1
        else:
            assert not res.ok
            assert res.failure.phase == "row"
            assert res.failure.row_index == 1
            assert res.failure.sqlstate == "23505"
    # Concurrent trial loads never mutated the published package.
    assert STORE.package(jid) == pkg_before
