import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the wine dataset
wine_df = pd.read_csv('winequality-red.csv')

# Create the predictor (X) and target (y) variables
X = wine_df.drop('quality', axis=1)
y = wine_df['quality'].apply(lambda yval: 1 if yval >= 7 else 0)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# Train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
# accuracy on test data
X_test_prediction = model.predict(X_test)
print(accuracy_score(X_test_prediction, Y_test))

# web app
st.title("Wine Quality Prediction Model")

# Input wine features separated by commas
input_text = st.text_input("Enter all Wine Features (comma separated)")

if st.button("Predict"):
    try:
        # Convert input string to list of floats
        input_text_list = [float(x) for x in input_text.split(',')]
        features = np.asarray(input_text_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("🍷 Good Quality Wine")
        else:
            st.error("🍷 Bad Quality Wine")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values separated by commas.")
