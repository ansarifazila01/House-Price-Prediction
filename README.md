# House Price Prediction – EstateIQ

## About the Project

I have created this project to predict house prices using Machine Learning. In this project, users can enter property details like property type, location, and area to get an estimated house price.

I have used Hyderabad house price data to train the model and created a simple web application using Streamlit.

## Features

- Predicts estimated house prices.
- Different property types like 1 BHK, 2 BHK, 3 BHK, Villas and Residential Plots.
- Select different locations.
- Enter property area in square feet.
- Simple and easy-to-use interface.

## Technologies Used

- **Python:** Used for writing the project code.
- **Pandas:** Used for handling and processing data.
- **NumPy:** Used for numerical operations.
- **Scikit-learn:** Used for building and training the Machine Learning model.
- **Random Forest Regressor:** Used for predicting house prices.
- **Streamlit:** Used to create the web application.
- **JSON:** Used to store the property dataset.
- **Pickle:** Used to save the trained model.

## Dataset

I have used a Hyderabad house price dataset containing property details such as:

- Property Type
- Location
- Area
- Price

## Machine Learning Model

I used the Random Forest Regressor algorithm to train the model.

The model takes property type, location, and area as input and predicts the estimated property price.

### Model Evaluation

- Mean Absolute Error (MAE): 28.35
- R² Score: 0.6784

## Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── data/
    ├── price.json
    └── house_model.pkl
```

## How to Run the Project

**Step 1: Clone the repository**

```bash
git clone https://github.com/ansarifazila01/House-Price-Prediction.git
```

**Step 2: Open the project folder**

```bash
cd House-Price-Prediction
```

**Step 3: Create a virtual environment**

```bash
python -m venv venv
```

**Step 4: Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**Step 5: Install the required libraries**

```bash
python -m pip install -r requirements.txt
```

**Step 6: Run the application**

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Note

This project provides estimated house prices based on the available Hyderabad property dataset. Actual property prices may vary.

## Project Name
**EstateIQ – House Price Prediction**.
