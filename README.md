# lssvr


`lssvr` is a Python module implementing the Least Squares Support Vector Regression using the scikit-learn as base.


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
