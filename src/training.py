# Import modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mymodule as mod

# Path to the CSV file containing the data.
DATA_PATH = "../data/data.csv"

# Check if the file exists.
# Check if data file is not empty.
# Check if parsing the CSV file is possible.
# Check if the file can be read.
try:
    df = pd.read_csv(DATA_PATH, header=0)
except FileNotFoundError:
    print(f"Error: File not found at path: {DATA_PATH}")
    raise SystemExit(1)
except IOError:
    print(f"Error: Can't read: {DATA_PATH}")
    raise SystemExit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The CSV file at path {DATA_PATH} is empty.")
    raise SystemExit(1)
except pd.errors.ParserError:
    print(
        f"Error: Unable to parse the CSV file at path {DATA_PATH}. Check the file format."
    )
    raise SystemExit(1)

# Data loaded, we can continue.
# Save the data in 2 numpy array.
km = df["km"].values
price = df["price"].values

# Plot the data.
# fig, ax = plt.subplots()
# ax.scatter(km, price)
# ax.set_xlabel("Mileage(km)")
# ax.set_ylabel("Price")
# ax.set_title("Scatter plot of the data")
# plt.savefig("../images/Scatter plot of the data.png")
# plt.show()

# Features normalization is a technique used to normalize the range of independent variables or features of data.
# Normalization prevents overflows in the gradient descent algorithm.
km_normalized = (km - np.mean(km)) / np.std(km)

# Plot of the normalized data.
# fig, ax = plt.subplots()
# ax.scatter(km_normalized, price)
# ax.set_xlabel("Mileage(km)")
# ax.set_ylabel("Price")
# ax.set_title("Scatter plot of the normalized data")
# plt.savefig("../images/Scatter plot of the normalized data.png")
# plt.show()

# Initialize the parameters.
# Leaning rate is adjusting the speed of the gradient descent.
# Number of iterations is the max number of times the gradient descent will be applied.
theta0 = 0.0
theta1 = 0.0
learning_rate = 0.01
num_iterations = 2000

# Store the cost for analysis and visualization.
costs = []

# Plot the data and model before training.
# fig, ax = plt.subplots()
# ax.scatter(km, price, label="Data")
# ax.plot(km, mod.model(theta0, theta1, km), color="red", label="Model")
# ax.set_xlabel("Mileage(km)")
# ax.set_ylabel("Price")
# ax.legend()
# ax.set_title("Linear Regression before training")
# plt.savefig("../images/Linear Regression before training.png")
# plt.show()

# Train the model with gradient descent.
for iteration in range(num_iterations):
    # Calculate the current cost.
    current_cost = mod.cost_function(km_normalized, price, theta0, theta1)
    costs.append(current_cost)

    # # Update the parameters with gradient descent.
    theta0, theta1 = mod.update_parameters(
        km_normalized, price, theta0, theta1, learning_rate
    )

    # # Stop the algorithm if the cost is not decreasing.
    if iteration > 0 and np.abs(costs[iteration - 1] - costs[iteration]) < 1e-6:
        break

# Print iterations and cost.
print(f"Iteration {iteration}, Cost: {current_cost}")

# Re-adjust theta0 and theta1 for unnormalized data.
theta0 = theta0 - (theta1 * np.mean(km) / np.std(km))
theta1 = theta1 / np.std(km)

# Print the final parameters.
print(f"Adjusted theta0 = {theta0}, Adjusted theta1 = {theta1}")

# Save the parameters in a CSV file.
pd.DataFrame({"theta0": [theta0], "theta1": [theta1]}).to_csv(
    "../data/parameters.csv", index=False
)
# Plot the data and the model after training.
# fig, ax = plt.subplots()
# ax.scatter(km, price, label="Data")
# ax.plot(km, mod.model(theta0, theta1, km), color="red", label="Model")
# ax.set_xlabel("Mileage(km)")
# ax.set_ylabel("Price")
# ax.legend()
# ax.set_title("Linear Regression after training")
# plt.savefig("../images/Linear Regression after training.png")
# plt.show()

# Plot the cost.
# fig, ax = plt.subplots()
# ax.plot(costs)
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Cost")
# ax.set_title("Cost function")
# plt.savefig("../images/cost.png")
# plt.show()

# ANIMATION PART
# Create a figure and axis for the animation.
fig, ax = plt.subplots()
# ax.axis([np.min(km) * 0.75, 250000, np.min(price) * 0.75, 9000])
ax.axis([0, 250000, -400, 9000])
ax.set_xlabel("Mileage(km)")
ax.set_ylabel("Price")
ax.set_title("Linear Regression Animation")

# Create an empty scatter plot for the data points.
scatter = ax.scatter(km, price, label="Data")

# Create an empty line plot for the model.
(line,) = ax.plot(km, mod.model(0, 0, km), "r", label="Model")

# Create an empty text for displaying the iteration and cost.
text = ax.text(
    0.02,
    0.95,
    f"Iteration: 0, Cost: {mod.cost_function(km_normalized, price, 0, 0):.2f}",
    transform=ax.transAxes,
)

theta0 = 0.0
theta1 = 0.0


# Update function for the animation.
def update(frame):
    global theta0, theta1

    # Calculate the current cost.
    current_cost = mod.cost_function(km_normalized, price, theta0, theta1)
    costs.append(current_cost)

    # Update the parameters with gradient descent.
    theta0, theta1 = mod.update_parameters(
        km_normalized, price, theta0, theta1, learning_rate
    )

    # Update the line plot with the model.
    theta0_unscaled = theta0 - (theta1 * np.mean(km) / np.std(km))
    theta1_unscaled = theta1 / np.std(km)
    line.set_data(km, mod.model(theta0_unscaled, theta1_unscaled, km))

    # Update the text with the iteration and cost.
    text.set_text(f"Iteration: {frame + 1}, Cost: {current_cost:.2f}")

    # Stop the animation if the cost is not decreasing.
    if frame > 0 and np.abs(costs[frame - 1] - costs[frame]) < 1e-6:
        animation.event_source.stop()


# Create the animation.
animation = animation.FuncAnimation(fig, update, frames=num_iterations, interval=5)

# Show the plot.
plt.legend()
plt.show()
