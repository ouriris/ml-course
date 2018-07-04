from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

boston = load_boston()
X = boston.data
y = boston.target
print(X.shape)
print(boston.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    liner_regression = LinearRegression(normalize=True)
    pipline = Pipeline([("polynomial_features", polynomial_features), ("liner_regression", liner_regression)])

    return pipline

model = polynomial_model(degree=2)
start = time.clock()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0: .6f}; train_score: {1:0.6f}; cv_score: {2: .6f}'.format(time.clock() - start, train_score, cv_score))

