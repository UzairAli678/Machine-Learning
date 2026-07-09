# Import Streamlit so we can build the web app interface.
import streamlit as st

# Import the reusable model helpers.
from house_price_model import predict_price, train_model


# Set the page title, icon, and layout for a cleaner UI.
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Show the main title of the application.
st.title("House Price Prediction with Linear Regression")

# Add a short description of what the app does.
st.write("Enter a house size in square feet and the model will estimate the price.")


# Cache the trained model so Streamlit does not retrain it on every interaction.
@st.cache_resource
def get_model():
    # Train the model once and keep it cached.
    return train_model()


# Load the model and the evaluation metrics.
model, metrics = get_model()

# Create a numeric input for the user to type the house size.
house_size = st.number_input("House size (sq ft)", min_value=300.0, max_value=5000.0, value=1600.0, step=50.0)

# Add a button so the prediction happens on demand.
if st.button("Predict price"):
    # Predict the price for the selected size.
    predicted_price = predict_price(model, house_size)

    # Show the predicted result.
    st.success(f"Estimated house price: ${predicted_price:,.2f}")

# Show the learned model values so the user can see how the line fits the data.
st.subheader("Model details")

# Display the slope of the regression line.
st.write(f"Coefficient: {model.coef_[0]:.2f}")

# Display the intercept of the regression line.
st.write(f"Intercept: {model.intercept_:.2f}")

# Display the model metrics in a compact layout.
col1, col2, col3 = st.columns(3)

# Show the MAE metric.
col1.metric("MAE", f"{metrics['mae']:.2f}")

# Show the MSE metric.
col2.metric("MSE", f"{metrics['mse']:.2f}")

# Show the R-squared metric.
col3.metric("R-squared", f"{metrics['r2']:.2f}")

# Add a note so the user understands the data is synthetic.
st.caption("This demo uses a small synthetic dataset, so it is for learning purposes only.")