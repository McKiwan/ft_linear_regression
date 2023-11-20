import numpy as np


# Hypothesis to calculate the price of a car.
def model(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


# Cost function that returns the mean squared error between the predicted prices and the actual prices of a car datase.
def cost_function(mileage, price, theta0, theta1):
    value = 0.0
    for i in range(len(mileage)):
        value += (model(theta0, theta1, mileage[i]) - price[i]) ** 2
    return value / (2 * len(mileage))


# Gradient descent algorithm for theta0 parameter.
def gradient_descent_theta0(mileage, price, theta0, theta1):
    value = 0.0
    for i in range(len(mileage)):
        value += model(theta0, theta1, mileage[i]) - price[i]
    return value / len(mileage)


# Gradient descent algorithm for theta1 parameter.
def gradient_descent_theta1(theta0, theta1, mileage, price):
    value = 0.0
    for i in range(len(mileage)):
        value += (model(theta0, theta1, mileage[i]) - price[i]) * mileage[i]
    return value / len(mileage)


# Update the parameters theta0 and theta1.
def update_parameters(mileage, price, theta0, theta1, learning_rate):
    new_theta0 = theta0 - learning_rate * gradient_descent_theta0(
        mileage, price, theta0, theta1
    )
    new_theta1 = theta1 - learning_rate * gradient_descent_theta1(
        theta0, theta1, mileage, price
    )
    return new_theta0, new_theta1
