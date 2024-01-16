#UG-POSMT

## Introduction

How to install libraries
`python -m pip install requirements.txt`

## How to run

`python src/main.py`

## Used packages
* spacy
* numpy
* pytorch
* jupyter notebook
* transformers
* loguru
* pandas

## Data

TODO

## Methods in MaskGenerator
There's 6 methods which were created by using 3 flags

### List of flags
* look_at_children: add you children to 'attention row'
* inherit_from_parent: inherit 'attention row' from your parent
* look_at_yourself: are you looking at yourself in attention mask

### List of methods
| name | look_at_children | inherit_from_parent | look_at_yourself |
| ------ | ----------- | ----------- | ----------- |
| shallow | False | False | True |
| deep | True | False | True |
| only_children | False | True | False |
| only_parent | True | False | True |
| full | True | True | True |
| full_without_yourself | True | True | False |
