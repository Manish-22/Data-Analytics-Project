from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import sqrt

def LRModel(Xtrain, Ytrain, Xtest, Ytest):

    Regressor = LinearRegression()
    model = Regressor.fit(Xtrain, Ytrain)
    predictedLR = Regressor.predict(Xtest)
    return (model, sqrt(metrics.mean_squared_error(Ytest, predictedLR)))