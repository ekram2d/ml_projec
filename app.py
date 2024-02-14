from flask import render_template, request, Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import math
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import t
from sklearn.metrics import mean_squared_error


app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('new.csv')

# Define features and target
X = dataset[['size', 'bed room', 'total bathroom', 'attached bathroom', 'kitchen',
             'varanda', 'dining room', 'drawing room', 'floor', 'lift', 'security', 'gasline', 'area', 'city']]
y = dataset['rent']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Define transformers for numeric and categorical features
numeric_features = ['size', 'bed room', 'total bathroom', 'attached bathroom', 'kitchen',
                    'varanda', 'dining room', 'drawing room', 'floor', 'lift', 'security', 'gasline']
numeric_transformer = SimpleImputer(strategy='mean')

categorical_features = ['area', 'city']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])


# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def calculate_prediction_interval(y_pred, rmse, confidence=0.50):
    df = len(y_pred) - 1
    std_error = rmse * np.sqrt(1 + 1/len(y_pred))
    t_statistic = t.ppf((1 + confidence) / 2, df=df)
    margin_of_error = t_statistic * std_error
    return margin_of_error


Mse = mean_squared_error(y_test, y_pred)
Rmse = math.sqrt(Mse)
margin_error = calculate_prediction_interval(y_pred, Rmse)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            'size': float(request.form['size']),
            'bed room': float(request.form['bedroom']),
            'total bathroom': float(request.form['bathroom']),
            'attached bathroom': float(request.form['attached_bathroom']),
            'kitchen': float(request.form['kitchen']),
            'varanda': float(request.form['varanda']),
            'dining room': float(request.form['dining_room']),
            'drawing room': float(request.form['drawing_room']),
            'floor': float(request.form['floor']),
            'lift': float(request.form['lift']),
            'security': float(request.form['security']),
            'gasline': float(request.form['gasline']),
            'city': request.form['city'],
            'area': request.form['area']
        }

        input_data = pd.DataFrame([features])
        prediction = model.predict(input_data)[0]

        lower_bound = round(prediction-margin_error)
        upper_bound = round(prediction+margin_error)

        rent = f"{lower_bound} - {upper_bound}"
        return render_template('index.html', lower_bound=lower_bound, upper_bound=upper_bound)

    except Exception as e:
        # print(e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
