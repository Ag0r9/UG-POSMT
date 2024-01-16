format:
	isort src/*.py
	black src tests

lang:
	python -m spacy download en_core_web_sm
	python -m spacy download fr_core_news_sm
	python -m spacy download es_core_news_sm
	python -m spacy download de_core_news_sm

mypy:
	python -m mypy src/ --ignore-missing-imports

tests:
	python -m pytest tests/
