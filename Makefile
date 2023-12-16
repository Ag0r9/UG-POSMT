format:
	isort src/*.py
	black src

lang:
	python -m spacy download en_core_web_sm

mypy:
	python -m mypy src/ --ignore-missing-imports