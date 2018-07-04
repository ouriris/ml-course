import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams

n_dots = 200

X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)

Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)


def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    liner_regression = LinearRegression(normalize=True)
    pipline = Pipeline([("polynomial_features", polynomial_features), ("liner_regression", liner_regression)])

    return pipline

degrees = [2, 3, 5, 10]
results = []

for degree in degrees:
    model = polynomial_model(degree=degree)
    model.fit(X, Y)
    train_score = model.score(X, Y)
    mse = mean_squared_error(Y, model.predict(X))
    results.append({"model": model, "degree": degree, "score": train_score, "mse": mse})

for result in results:
    print('degree: {}; train_score: {}; mse: {}'.format( result["degree"], result["score"], result["mse"]))


plt.figure(figsize = (12, 6), dpi = 200, subplotpars=SubplotParams(hspace=0.3))
for i, result in enumerate(results):
    fig = plt.subplot(2, 2, i + 1)
    plt.xlim(-8, 8)
    plt.title("liner regression degree={}".format(result["degree"]))
    plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, result["model"].predict(X), 'r-')

plt.show()
