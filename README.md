# PI-DimRe4RFs-FieldReducer

## Overview
This repository implements the PI-DimRe4RFs-FieldReducer, which includes a complete pipeline for generating and training neural network surrogates for reduced order modeling of materials using random fields.

## Pipeline Phases
1. Generate training samples of Young's modulus with given Matern parameters.
2. Precompute points in the reduced space and fit surrogate models.
3. Train dimension reducer.
4. Evaluate on test datasets.

## Requirements
Install the required dependencies:
```bash
git clone https://github.com/LigneMaginotLYF/PI-DimRe4RFs-FieldReducer
cd PI-DimRe4RFs-FieldReducer
pip install -r requirements.txt
```
