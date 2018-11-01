import numpy as np
from lssvr import LSSVR, RegENN_LSSVR, RegCNN_LSSVR, DiscENN_LSSVR, MI_LSSVR, AM_LSSVR

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

model = LSSVR()
model.fit(X_train, y_train, kernel='linear')
y_hat = model.predict(X_test)
print('LSSVR\nMSE', mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))

model = RegENN_LSSVR()
model.fit(X_train, y_train, kernel='rbf', alpha=1, n_neighbors=9)
y_hat = model.predict(X_test)
print('\nRegENN_LSSVR\nMSE', mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))

model = RegCNN_LSSVR()
model.fit(X_train, y_train, kernel='rbf', alpha=1, n_neighbors=9)
y_hat = model.predict(X_test)
print('\nRegCNN_LSSVR\nMSE', mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))

model = DiscENN_LSSVR()
model.fit(X_train, y_train, kernel='rbf', n_neighbors=2)
y_hat = model.predict(X_test)
print('\nDiscENN\nMSE', mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))

model = MI_LSSVR()
model.fit(X_train, y_train, kernel='linear', alpha=0.35, n_neighbors=6)
y_hat = model.predict(X_test)
print('\nMI_LSSVR - alpha = {}\nMSE'.format(0.35), mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))

model = AM_LSSVR()
model.fit(X_train, y_train, kernel='linear')
y_hat = model.predict(X_test)
print('\nAM_LSSVR \nMSE', mean_squared_error(y_test, y_hat))
print('R2 ',model.score(X_test, y_test))
print('SP ', 1 - len(model.supportVectorLabels)/len(y_train))