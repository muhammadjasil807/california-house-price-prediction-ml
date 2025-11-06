California House Price Prediction using Machine Learning

This project predicts house prices in California using a Linear Regression model.
It uses the California Housing dataset from Scikit-learn, applies feature scaling, trains the model, evaluates its performance, and visualizes the results.

Overview
The California Housing dataset contains data from the 1990 California census.
It includes 20,640 samples with 8 numerical features describing districts:

- MedInc: median income
- HouseAge: median house age
- AveRooms: average rooms per household
- AveBedrms: average bedrooms per household
- Population: total population per district
- AveOccup: average household occupancy
- Latitude: geographic coordinate
- Longitude: geographic coordinate

The goal is to predict the median house value (PRICE) for each district.

Features
- Clean and well-structured machine learning workflow
- Uses built-in California Housing dataset from scikit-learn
- Demonstrates feature scaling, model training, and evaluation
- Visualizes data distributions, correlations, and predictions
- Beginner-friendly and great for portfolio demonstration

Tech Stack
Programming Language: Python 3
Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
Algorithm: Linear Regression
Dataset: California Housing dataset from scikit-learn

Project Structure
california-house-price-prediction-ml
│
├── housing_notebook.ipynb       (Colab notebook with full code)
├── model.pkl                     (Saved Linear Regression model)
├── scaler.pkl                    (Saved StandardScaler)
├── requirements.txt              (Dependencies)
└── README.md                     (Project documentation)

Installation and Running
Step 1 — Clone the Repository
git clone https://github.com/YOUR-USERNAME/california-house-price-prediction-ml.git
cd california-house-price-prediction-ml

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Run the Notebook
Open the Jupyter or Google Colab notebook and execute each cell to train and test the model.

Model Details
Dataset: California Housing (20,640 samples, 8 features)
Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
Algorithm: Linear Regression
Target Variable: PRICE
Evaluation Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score

Example Output
Mean Squared Error: 0.54
Root Mean Squared Error: 0.73
R² Score: 0.61

Future Improvements
- Try more advanced regression algorithms like Random Forest Regressor or XGBoost
- Hyperparameter tuning for better performance
- Deploy on Streamlit for interactive house price predictions
- Include more feature engineering for improved accuracy

Conclusion
The California House Price Prediction project demonstrates how machine learning can model continuous numerical data and make accurate predictions.
It’s a fundamental step in understanding regression tasks and real-world ML applications.
