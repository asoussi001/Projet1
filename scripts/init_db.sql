-- Initialisation PostgreSQL — Tech Office Cockpit

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS document_index (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    project_key     TEXT,
    qdrant_collection TEXT NOT NULL,
    chunk_count     INTEGER DEFAULT 0,
    embedding_model TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    checksum        TEXT NOT NULL UNIQUE,
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key       TEXT UNIQUE NOT NULL,
    analysis_type   TEXT NOT NULL,
    result          JSONB NOT NULL,
    llm_model       TEXT NOT NULL,
    tokens_used     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS analysis_audit (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_key       TEXT NOT NULL,
    analysis_type   TEXT NOT NULL,
    triggered_by    TEXT NOT NULL,
    status          TEXT NOT NULL,
    duration_ms     INTEGER,
    tokens_used     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_audit_issue ON analysis_audit(issue_key);
CREATE INDEX IF NOT EXISTS idx_analysis_audit_created ON analysis_audit(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_index_project ON document_index(project_key);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires ON analysis_cache(expires_at);
