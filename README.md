## From Notebook to package 🎁

It is time to move away from Jupyter Notebook, and start writing reusable code with python packages, modules and classes.

In the exercice, nothing new, simply copy pasting functions we already implemented yesterday, and organize them inside different modules.

### Package structure 🗺

We have created for you the following structure:

```bash
├── TaxiFareModel
│   ├── __init__.py   # to make it a python package
│   ├── data.py       # functions to get and clean data
│   ├── encoders.py   # your custom encoders and transformers for our Pipeline
│   ├── trainer.py    # utility functions
│   └── utils.py      # containing main class that will run our Pipeline
```

### Setup your package ⚙️

You are going to create a package from your pipeline. To achieve this, we provide you a minimal template generator package [`packgenlite`](https://github.com/krokrob/packgenlite).

- Install `packgenlite` package from Github

```bash
pip install git+https://github.com/krokrob/packgenlite.git
```

- Create a new project `TaxiFareModel` in your working directory

```bash
cd ~/code/<user.github_nickname>
packgenlite TaxiFareModel
```

- Copy the code we provide into your project

<details>
<summary>💡 Hint
</summary>

```bash
cp -r ~/code/<user.github_nickname>/data-challenges/07-Data-Engineering/02-ML-Iteration/03-Notebook-to-package/TaxiFareModel ~/code/<user.github_nickname>/TaxiFareModel/
```

</details>

- Add a `.gitignore` file to your project with this folders to be ignored:

```txt
__pycache__/
*.egg-info/
.ipynb_checkpoints
```

- Add a `requirements.txt` file with the list of the python package you need to run your pipeline

```txt
numpy
pandas
scikit-learn
```

- Add a `setup.py` file with the following configuration to be able to install your package locally

```python
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'pandas==1.1.3',
    'scikit-learn==0.23.2'
]

setup(
    name='TaxiFareModel',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline'
)
```

- Make sure your package has the following structure
```bash
.
├── TaxiFareModel
│   ├── __init__.py
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── setup.py
```

- Keep tracks of your changes with `git`

```bash
git init
git add TaxiFareModel/ .gitignore requirements.txt setup.py
git commit -m 'start TaxiFareModel package'
```

👍 Your package is ready to be implemented!

### Implement your package 🛠

#### `data.py`

Inspect the functions `get_data` and `clean_data` that are already given to you.

_NB: Here we provide you with the same functions so that we all get and clean data the same way._

#### `utils.py`

In `utils.py` this is where you can have :
- `haversine_distance` method
- `compute_rmse` method

#### `encoders.py`

In `encoders.py` let's put the custom encoders and transformers you have for distance and time features

**Reminder**

Let us be clear about the use of `encoders.py` here:
- It contains all custom pipeline preprocessing blocs not provided by sklearn
- These preprocessing blocks are the DistanceTransformer and the TimeFeaturesEncoder

#### `trainer.py`

Implement the main class here.

The `Trainer` class is the main class. It should have:
- a `__init__()` method instanciating class
- a `set_pipeline()` method that builds the pipeline
- a `run()` method that trains the pipeline
- a `evaluate()` method evaluating model

Make sure you are confident with following notions:
- attributes and methods of a class
- `**kwargs` argument of a function and how to use it, (help [HERE](https://www.programiz.com/python-programming/args-and-kwargs) if unclear)

```python
class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

    def run(self):
        """set and train the pipeline"""

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
```

#### Test your package!

Once you have everything setup, test that it works by running:

```bash
python -m TaxiFareModel.trainer
```

Or

```bash
python -i TaxiFareModel/trainer.py
```

#### Hints for debugging 🐛
- Do not hesitate to only breakdown your code into smaller calls for debugging
- Use [if \_\_name__ == '\_\_main__'](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/) at the end of each `.py` file to debug
- For instance to debug data.py, add at the end:

```python
# TaxiFareModel/data.py

if __name__ == '__main__':
    df = get_data()
```

Then in a `ipython` session from your terminal, you can run:

```bash
In [1]: %run TaxiFareModel/data.py

In [2]: df.shape
```

### Install your package

When your package is all set, you would like to install it locally so that it can be imported anywhere. From your package folder, run:

```bash
pip install -e .
```

Then you can open the `taxi_fare_model_package_testing.ipynb` and run the cells.

### Push your package on GitHub 🐙😸

1. Create a repository on GitHub
2. Add a remote between your package and the GitHub repository
3. Push your code on the GitHub repository

👍 Good job!
