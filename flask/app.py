import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
        # load the data into a pandas DataFrame
    df = pd.read_excel('C:/Users/medal/OneDrive/Bureau/freshwaterglobal.xlsx')

    # separate the target variable (Emissions) from the features
    X = df[['date', 'country']]
    y = df['Annual freshwater withdrawals']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create a one-hot encoding of the Country column
    X_train_encoded = pd.get_dummies(X_train, columns=['country'])
    X_test_encoded = pd.get_dummies(X_test, columns=['country'])

    # ensure that the training and testing sets have the same number of columns
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for c in missing_cols:
        X_test_encoded[c] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    # create an XGBoost regressor object and fit it to the training data
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train_encoded, y_train)

    # get the input data from the client
    data = request.get_json()

    # extract the target country and date from the input data
    target_country = data['country']
    target_date = int(data['date'])

    # create a new DataFrame with the target country and date
    target_df = pd.DataFrame({
        'date': [target_date],
        'country': [target_country]
    })

    # encode the target DataFrame using the same one-hot encoding as the training data
    target_df_encoded = pd.get_dummies(target_df, columns=['country'])
    missing_cols = set(X_train_encoded.columns) - set(target_df_encoded.columns)
    for c in missing_cols:
        target_df_encoded[c] = 0
    target_df_encoded = target_df_encoded[X_train_encoded.columns]

    # use the trained model to make a prediction
    predicted_Annualfreshwaterwithdrawals = xgb_model.predict(target_df_encoded)

    # return the predicted value as a JSON object
    return jsonify({'prediction': predicted_Annualfreshwaterwithdrawals.tolist()})


@app.route('/distinct_countries', methods=['GET'])
def get_distinct_countries():
    df = pd.read_excel('C:/Users/medal/OneDrive/Bureau/freshwaterglobal.xlsx')
    # Get the distinct countries
    distinct_countries = list(df['country'].unique())

    # Return the list as a JSON response
    return jsonify(distinct_countries)






@app.route('/predictpoverty', methods=['POST'])
def predictpoverty():
        
    # load the data into a pandas DataFrame
    df = pd.read_excel('C:/Users/medal/OneDrive/Bureau/povertyline.xlsx')

    # separate the target variable (Emissions) from the features
    X = df[['Country Name', 'Year']]
    y = df['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)']
    if np.isnan(y).any() or np.isinf(y).any():
        # Replace invalid values with the mean of the label array
        y[np.isnan(y) | np.isinf(y)] = np.mean(y[np.isfinite(y)])
        # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create a one-hot encoding of the Country column
    # create a one-hot encoding of the Country column
    X_train_encoded = pd.get_dummies(X_train, columns=['Country Name'])
    X_test_encoded = pd.get_dummies(X_test, columns=['Country Name'])

    # ensure that the training and testing sets have the same number of columns
    # ensure that the training and testing sets have the same number of columns
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for c in missing_cols:
        X_test_encoded[c] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    # create an XGBoost regressor object and fit it to the training data
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train_encoded, y_train)


    # get the input data from the client
    data = request.get_json()

    # extract the target country and date from the input data
    target_country = data['country']
    target_date = int(data['date'])

    # create a new DataFrame with the target country and date
    target_df = pd.DataFrame({
        'Year': [target_date],
        'Country Name': [target_country]
    })

    # encode the target DataFrame using the same one-hot encoding as the training data
    target_df_encoded = pd.get_dummies(target_df, columns=['Country Name'])
    missing_cols = set(X_train_encoded.columns) - set(target_df_encoded.columns)
    for c in missing_cols:
        target_df_encoded[c] = 0
    target_df_encoded = target_df_encoded[X_train_encoded.columns]

    # use the trained model to make a prediction
    predicted_poverty = xgb_model.predict(target_df_encoded)

    # return the predicted value as a JSON object
    return jsonify({'prediction': predicted_poverty.tolist()})





# create a route for the prediction endpoint
@app.route('/predict_hunger', methods=['POST'])
def predict_hunger():
    
    # load the data into a pandas DataFrame
    df = pd.read_excel('C:/Users/medal/OneDrive/Bureau/AGRICULTURE_F.xlsx')
    df.dropna(inplace=True)
    # separate the target variable (Taux de famine) from the features
    X = df[['Date', 'Country']]
    y = df['Taux de famine']

    # create a one-hot encoding of the Country column
    X_encoded = pd.get_dummies(X, columns=['Country'])

    # create a RandomForestRegressor object and fit it to the data
    rf = RandomForestRegressor()
    rf.fit(X_encoded, y)

    # get the request data as a JSON object
    req = request.get_json()
    
    # extract the target_country and target_date from the request data
  
    
    target_country = request.json['country']
    target_date = int(request.json['date'])
    # create a DataFrame with the target country and date
    target_df = pd.DataFrame({
        'Date': [target_date],
        'Country': [target_country]
    })
    
    # encode the target DataFrame using the same one-hot encoding as the training data
    target_df_encoded = pd.get_dummies(target_df, columns=['Country'])
    missing_cols = set(X_encoded.columns) - set(target_df_encoded.columns)
    for c in missing_cols:
        target_df_encoded[c] = 0
    target_df_encoded = target_df_encoded[X_encoded.columns]

    # use the trained model to make a prediction
    predicted_value = rf.predict(target_df_encoded)

    # return the predicted value as a JSON object
    return jsonify({'prediction': predicted_value.tolist()})

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)

