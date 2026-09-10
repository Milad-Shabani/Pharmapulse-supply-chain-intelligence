.PHONY: install data pipeline test lint clean all

install:
	pip install -r requirements.txt

data:
	python scripts/generate_sample_data.py

pipeline:
	python scripts/run_pipeline.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

all: install data pipeline test

clean:
	rm -rf data/processed/*.parquet
	rm -rf reports/*.xlsx reports/*.html reports/*.csv
	find . -name "__pycache__" -type d -exec rm -rf {} +
