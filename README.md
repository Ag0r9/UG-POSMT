#UG-POSMT

## Introduction

`python -m pip install requirements.txt`

W requirements.txt jest pytorch cpu.

Lubię używać isort i black do formatu.
Jak chcecie używać to piszcie `make format`

## TODO:
* Przygotowanie danych treningowych i testowych - JK
* stworzenie funkcji generującej self-attention mask - AG
* stworzenie funkcji, która nakłada maskę na model - PH
* wybranie co najmniej jednego języka do tłumaczenia
    * Eng -> Ger
* możliwość generowania tłumaczenia z jednego języka na drugi

## Używane biblioteki
Jakby kiedyś requirements.txt się zepsuło
* spacy
* numpy
* pytorch
* jupyter notebook
* transformers
* loguru
* pandas

## Zarządzanie danymi
Nie będziemy zapisywać na repo wszystkich danych (nie zmieszczą się pewnie).
Moim pomysłem jest zapisać je gdzieś w chumrze i zostawić linki w txt lub yml

Przykład:
* raw: drive.google.com/blablabla
* preprocessed: drive.google.com/blebleble
