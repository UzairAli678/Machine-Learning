# Import the shared model helpers used by both the command-line script and the Streamlit app.
from house_price_model import predict_price, train_model


# Train the model and compute evaluation metrics.
model, metrics = train_model()

# Predict the price for a sample house size.
predicted_price = predict_price(model, 1600)

# Print the slope of the learned regression line.
print(f"Coefficient: {model.coef_[0]:.2f}")

# Print the intercept of the learned regression line.
print(f"Intercept: {model.intercept_:.2f}")

# Print the mean absolute error.
print(f"Mean Absolute Error: {metrics['mae']:.2f}")

# Print the mean squared error.
print(f"Mean Squared Error: {metrics['mse']:.2f}")

# Print the R-squared value.
print(f"R-squared: {metrics['r2']:.2f}")

# Print the predicted price for the sample house.
print(f"Predicted price for 1600 sq ft house: {predicted_price:.2f}")
