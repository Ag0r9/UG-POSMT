format:
	isort src/*.py
	black src tests

lang:
	python -m spacy download en_core_web_sm

mypy:
	python -m mypy src/ --ignore-missing-imports

tests:
	python -m pytest tests/
