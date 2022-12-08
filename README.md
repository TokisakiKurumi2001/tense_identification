# Tense identification

Tense identification using RoBERTa model.

Install the conda environment before install from this repo.

[Github repo](https://github.com/TokisakiKurumi2001/tense_identification)

## Tense transformation

Tense transformation take advantages of tense identification and other libraries to perform transforming tense of a given sentence.

**Repo**: [tense_transformation](https://github.com/TokisakiKurumi2001/tense_transformation)

**NOTE**: Just a toy repo I try to deploy on PyPI.

## How to run this repo

1. Create environment

```bash
conda create -f environment.yml
```

2. Install TeXid library

```bash
pip install -e .
```

Or you can install from PyPI (**not recommend**)
```bash
pip install TeXid
```

3. Train the model

```bash
python train.py
```

**NOTE**: Under no circumstance am I responsile for getting you into trouble while running the code. Just check the code before running it. Cheers and have a good day.
