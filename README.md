# lssvr

`lssvr` is a Python module implementing the [Least Squares Support Vector Regression][1] using the scikit-learn as base.

## instalation
the `lssvr` package is available in [PyPI](https://pypi.org/project/lssvr/). to install, simply type the following command:
```
pip install lssvr
```
or using [Poetry](python-poetry.org/):
```
poetry add lssvr
```

## basic usage

Example:

```Python
import numpy as np
from lssvr import LSSVR

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

model = LSSVR(kernel='rbf', gamma=0.01)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('MSE', mean_squared_error(y_test, y_hat))
print('R2 Score',model.score(X_test, y_test))
```


## contributing

this project is open for contributions. here are some of the ways for you to contribute:

 - bug reports/fix
 - features requests
 - use-case demonstrations

to make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a pull request!


[1]: https://doi.org/10.1016/S0925-2312(01)00644-0