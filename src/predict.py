import pandas as pd
import mymodule as mod

# Path to the CSV file containing the parameters.
PARAMETERS_PATH = "../data/parameters.csv"

try:
    thetas = pd.read_csv(PARAMETERS_PATH, header=0)
except FileNotFoundError:
    print(f"Error: File not found at path: {PARAMETERS_PATH}")
    raise SystemExit(1)
except IOError:
    print(f"Error: Can't read: {PARAMETERS_PATH}")
    raise SystemExit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The CSV file at path {PARAMETERS_PATH} is empty.")
    raise SystemExit(1)

# Data loaded, we can continue.
# Save the data in 2 distinct variables.
theta0 = thetas["theta0"].values[0]
theta1 = thetas["theta1"].values[0]

# Ask the user to enter a mileage.
while True:
    mileage = input("Please enter you car's mileage: ")
    try:
        mileage = int(mileage)
    except:
        print("Please input correct mileage.")
        continue
    if mileage < 0:
        print("Mileage should be positive, obviously.")
        continue
    else:
        print(f"Your car's mileage is {mileage} km.")
        break

# Predict the price of the car.
predicted_price = mod.model(theta0, theta1, mileage)
if predicted_price < 0:
    predicted_price = 0
print(f"The predicted price of your car is {predicted_price:.2f}$.")
