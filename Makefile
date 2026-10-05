install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	pytest -q

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

benchmark:
	python scripts/benchmark.py --url http://localhost:8000 --requests 32 --concurrency 8

smoke:
	python scripts/smoke_test.py --url http://localhost:8000
