CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    applicant_id VARCHAR(100) NOT NULL,
    default_probability NUMERIC(10, 6) NOT NULL,
    predicted_default BOOLEAN NOT NULL,
    risk_band VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    applicant_id VARCHAR(100) NOT NULL,
    default_probability NUMERIC(10, 6) NOT NULL,
    predicted_default BOOLEAN NOT NULL,
    risk_band VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_features JSONB,
    actual_default BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);