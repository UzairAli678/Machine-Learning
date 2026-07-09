# Linear Regression House Price Prediction

This folder contains a simple, fully commented example of house price prediction using linear regression, plus a Streamlit GUI where you can type a house size and get a predicted price.

## Files

- `house_price_model.py` - shared training and prediction logic.
- `house_price_prediction.py` - command-line example that prints the model output.
- `streamlit_app.py` - Streamlit GUI for entering house size and predicting price.

## What the script does

1. Creates a tiny example dataset with house area as the input feature.
2. Splits the data into training and testing sets.
3. Trains a `LinearRegression` model from scikit-learn.
4. Evaluates the model with MAE, MSE, and R-squared.
5. Predicts the price of a new 1600 sq ft house.
6. Lets you enter any house size in the Streamlit app and see the predicted price.

## How to run

Install the required package if needed:

```bash
pip install numpy scikit-learn streamlit
```

Run the script from this folder:

```bash
python house_price_prediction.py
```

Run the Streamlit app from this folder:

```bash
streamlit run streamlit_app.py
```

## Notes

- The dataset is synthetic and kept very small so the example is easy to understand.
- Every important line in the Python script is commented to explain what it does.
- The Streamlit app is the GUI version for interactive house-size input.
