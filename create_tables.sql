CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    started_at_ms FLOAT NOT NULL,
    ended_at_ms FLOAT NOT NULL
);

CREATE TABLE attempts (
    id TEXT PRIMARY KEY,
    config JSONB NOT NULL,
    usage JSONB NOT NULL,
    challenge JSONB NOT NULL,
    messages JSONB[] NOT NULL,
    python_code_str TEXT,
    train_attempts JSONB[] NOT NULL,
    test_attempt JSONB NOT NULL,
    fixing_id TEXT,
    run_id TEXT REFERENCES runs(id),
    train_accuracy FLOAT,
    test_accuracy FLOAT,
    avg_cell_diff_percent FLOAT,
    cost_cents FLOAT,
    fixing_ids TEXT[]
);