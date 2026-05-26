# firstML

A simple machine learning project that predicts solar cell efficiency (`PCE`) using photovoltaic performance features.

## Project structure

- `ML_Ready_Dataset.csv` - dataset containing `PCE`, `VOC`, `JSC`, and `FF`
- `requirements.txt` - Python package dependencies
- `setup_env.sh` - script to create a virtual environment and install requirements
- `preveiw.py` - generates scatter/regression plots for `PCE` vs `VOC`, `FF`, and `JSC`
- `Model.py` - trains a basic linear regression model to predict `PCE`
- `random_forest.py` - trains a random forest regressor and saves evaluation plots
- `finalRF.py` - alternative random forest script with feature importance and example prediction

## Setup

Use the included script to create a virtual environment and install dependencies.

```bash
bash setup_env.sh
source .venv/bin/activate
```

## Run the project

Generate visualization plots:

```bash
python preveiw.py
```

Train the linear regression model:

```bash
python Model.py
```

Train the random forest model and save plots:

```bash
python random_forest.py
```

## Notes

- The repo includes a `.gitignore` file that excludes `.venv/`
- The dataset is expected in the repository root as `ML_Ready_Dataset.csv`
- Install `scikit-learn` and the other required packages via `requirements.txt`
