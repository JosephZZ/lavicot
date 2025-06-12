.PHONY: help install install-dev clean test test-fast lint format type-check docs build upload

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,docs,benchmarks]"
	pre-commit install

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:  ## Run all tests with coverage
	pytest --cov=src/lavicot --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without slow tests
	pytest -m "not slow" --cov=src/lavicot --cov-report=term-missing

test-unit:  ## Run only unit tests
	pytest tests/unit/ --cov=src/lavicot --cov-report=term-missing

test-integration:  ## Run only integration tests
	pytest tests/integration/ --cov=src/lavicot --cov-report=term-missing

lint:  ## Run all linting checks
	flake8 src/ tests/ scripts/
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

format:  ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

type-check:  ## Run type checking with mypy
	mypy src/lavicot

quality:  ## Run all quality checks (lint, type-check, test)
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test-fast

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build:  ## Build package
	python -m build

upload:  ## Upload package to PyPI (requires authentication)
	twine upload dist/*

upload-test:  ## Upload package to TestPyPI
	twine upload --repository testpypi dist/*

train-debug:  ## Run debug training
	python scripts/train.py --config src/lavicot/config/defaults/debug.yaml

train:  ## Run full training
	python scripts/train.py --config src/lavicot/config/defaults/default.yaml

evaluate:  ## Run evaluation
	python scripts/evaluate.py --config src/lavicot/config/defaults/default.yaml

setup-env:  ## Setup development environment
	python -m venv venv
	source venv/bin/activate && pip install --upgrade pip
	source venv/bin/activate && $(MAKE) install-dev
	@echo "Environment setup complete. Activate with: source venv/bin/activate"

docker-build:  ## Build Docker image
	docker build -t lavicot:latest .

docker-run:  ## Run Docker container
	docker run --gpus all -it --rm -v $(PWD):/workspace lavicot:latest

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files 