"""Batch trial-load of incremental rows against a published package.

For every trial-load request a fresh isolated database is created on the
verification cluster, the job's *original* SQL package is loaded verbatim,
and the submitted rows are applied in submission order inside a single
transaction:

* immediate constraints (NOT NULL, PK/UNIQUE, NOT DEFERRABLE foreign keys,
  integer type bounds) reject the offending row right away;
* DEFERRABLE INITIALLY DEFERRED foreign keys are only checked at COMMIT.

Any failure rolls the whole transaction back — no partial batch is ever left
behind — and the isolated database is always discarded afterwards. The
published package is never mutated, downloads are unaffected, concurrent
trial loads of the same job each get their own database (no cross-batch
interference), and the source database is never touched at all.
"""
from __future__ import annotations

import secrets

import psycopg

from .config import settings
from .models import (
    TableSchema,
    TrialFailure,
    TrialLoadResponse,
    TrialRow,
)
from .sqlgen import q
from .verifier import _cleanup_database, _replace_dbname

#: Package files, loaded in the documented order, exactly as a user would.
PACKAGE_FILES = ("01_schema.sql", "02_data.sql", "03_constraints.sql")


def validate_trial_rows(
    selected: list[str],
    schemas: dict[str, TableSchema],
    rows: list[TrialRow],
) -> list[tuple[str, str]]:
    """Request-level checks: known tables and an exact per-row column set."""
    problems: list[tuple[str, str]] = []
    for i, row in enumerate(rows):
        ts = schemas.get(row.table)
        if ts is None or row.table not in selected:
            problems.append(
                (
                    f"rows[{i}].table",
                    f"表 {row.table!r} 不在该作业的所选表 {sorted(selected)} 中",
                )
            )
            continue
        expected = {c.name for c in ts.columns}
        given = set(row.values)
        missing = sorted(expected - given)
        extra = sorted(given - expected)
        if missing:
            problems.append((f"rows[{i}].values", f"缺少列值: {', '.join(missing)}"))
        if extra:
            problems.append(
                (f"rows[{i}].values", f"表中不存在这些列: {', '.join(extra)}")
            )
    return problems


def _insert_sql(schema: str, ts: TableSchema) -> str:
    cols = ", ".join(q(c.name) for c in ts.columns)
    placeholders = ", ".join(["%s"] * len(ts.columns))
    return f"INSERT INTO {q(schema)}.{q(ts.name)} ({cols}) VALUES ({placeholders})"


def _failure(
    exc: psycopg.Error, phase: str, row_index: int | None = None, table: str | None = None
) -> TrialFailure:
    diag = exc.diag
    return TrialFailure(
        phase=phase,
        row_index=row_index,
        table=table,
        sqlstate=exc.sqlstate or "?????",
        constraint=diag.constraint_name,
        message=diag.message_primary or str(exc),
    )


def run_trial_load(
    job_id: str,
    schema: str,
    selected: list[str],
    schemas: dict[str, TableSchema],
    package: dict[str, str],
    rows: list[TrialRow],
) -> TrialLoadResponse:
    """Load the original package into a fresh isolated DB and trial-apply rows.

    The whole batch runs in one transaction; the outcome reports either the
    per-table row counts after the trial load, or the database-reported
    failure (row phase or commit phase). Nothing of a failed batch survives.
    """
    db_name = f"{settings.verify_db_prefix}trial_{secrets.token_hex(8)}"
    try:
        with psycopg.connect(settings.verify_dsn, autocommit=True, connect_timeout=10) as adm:
            with adm.cursor() as cur:
                cur.execute(f"CREATE DATABASE {q(db_name)}")
        dsn = _replace_dbname(settings.verify_dsn, db_name)
        with psycopg.connect(dsn, autocommit=False, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {q(schema)}")
                conn.commit()
                # The published package is loaded verbatim, every time, so
                # every trial load starts from exactly the same baseline.
                for fname in PACKAGE_FILES:
                    cur.execute(package[fname])
                    conn.commit()

                # Single transaction for the whole ordered batch.
                for i, row in enumerate(rows):
                    ts = schemas[row.table]
                    params = [row.values[c.name] for c in ts.columns]
                    try:
                        cur.execute(_insert_sql(schema, ts), params)
                    except psycopg.Error as exc:
                        conn.rollback()  # nothing of the batch survives
                        return TrialLoadResponse(
                            job_id=job_id,
                            ok=False,
                            failure=_failure(exc, "row", i, row.table),
                        )
                try:
                    conn.commit()  # deferred foreign keys are checked here
                except psycopg.Error as exc:
                    conn.rollback()
                    return TrialLoadResponse(
                        job_id=job_id,
                        ok=False,
                        failure=_failure(exc, "commit"),
                    )

                counts: dict[str, int] = {}
                for t in selected:
                    cur.execute(f"SELECT count(*) FROM {q(schema)}.{q(t)}")
                    (counts[t],) = cur.fetchone()
                conn.commit()
                return TrialLoadResponse(job_id=job_id, ok=True, row_counts=counts)
    finally:
        # The scratch database is always released; the verdict above is
        # already decided and unaffected by cleanup.
        _cleanup_database(settings.verify_dsn, db_name, [])
