import streamlit as st
import pickle
import numpy as np
#import plotly.graph_objects as go

# Load the trained classifier
model = pickle.load(open('DTmodel.pkl', 'rb'))

def main():
    st.title("Iris Flower Classification")

    # Get user input for SepalLengthCm
    sepal_length = st.text_input("Enter Sepal Length (cm): ")
    
    # Get user input for SepalWidthCm
    sepal_width = st.text_input("Enter Sepal Width (cm): ")
    
    # Get user input for PetalLengthCm
    petal_length = st.text_input("Enter Petal Length (cm): ")
    
    # Get user input for PetalWidthCm
    petal_width = st.text_input("Enter Petal Width (cm): ")

    if st.button("Predict"):
        try:
            # Validate and convert user input to float
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)

            # Make a prediction using the trained model
            inputs = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
            prediction = model.predict(inputs)

            st.markdown(f"<h2 style='color: red'>{'Predicted class:'} {prediction[0]}</h2>", unsafe_allow_html=True)

            if prediction == 'Iris-setosa':
                st.image('i1.jpg')
            elif prediction == 'Iris-virginica':
                st.image("i2.jpeg")
            else:
                st.image("i3.jpeg")
        except ValueError:
            st.error("Please enter valid numeric values for the features.")

    st.write('Project by SURAJ SANAP')

if __name__ == '__main__':
    main()
