# Import NumPy so we can store the training data.
import numpy as np

# Import the linear regression model from scikit-learn.
from sklearn.linear_model import LinearRegression

# Import metrics so we can measure model quality.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the helper that splits data into training and testing sets.
from sklearn.model_selection import train_test_split


# Create the house-size values in square feet.
HOUSE_SIZES = np.array([600, 750, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300])

# Create the matching house prices in dollars.
HOUSE_PRICES = np.array([150000, 175000, 200000, 240000, 280000, 320000, 360000, 400000, 440000, 480000])


# Return the feature matrix and target values used by the model.
def load_dataset():
    # Reshape the sizes into a 2D array because scikit-learn expects that format.
    features = HOUSE_SIZES.reshape(-1, 1)

    # Return the features and target values.
    return features, HOUSE_PRICES


# Train the linear regression model and calculate evaluation metrics.
def train_model():
    # Load the feature matrix and target values.
    features, targets = load_dataset()

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Create the regression model.
    model = LinearRegression()

    # Fit the model on the training data.
    model.fit(X_train, y_train)

    # Predict the prices for the test
    predictions = model.predict(X_test)

    # Compute the evaluation metrics.
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Return the trained model and the metric values.
    return model, {"mae": mae, "mse": mse, "r2": r2}


# Predict the price for a single house size
def predict_price(model, house_size):
    # Put the user-entered size into the 2D shape expected by scikit-learn.
    input_value = np.array([[house_size]])

    # Predict the house price.
    predicted_price = model.predict(input_value)

    # Return the first and only predicted value as a normal float.
    return float(predicted_price[0])