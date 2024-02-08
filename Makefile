format:
	poetry run isort src/*.py
	poetry run black src tests

lang:
	poetry run python -m spacy download en_core_web_sm
	poetry run python -m spacy download fr_core_news_sm
	poetry run python -m spacy download es_core_news_sm
	poetry run python -m spacy download de_core_news_sm

mypy:
	poetry run python -m mypy src/ --ignore-missing-imports

test:
	poetry run pytest tests/
