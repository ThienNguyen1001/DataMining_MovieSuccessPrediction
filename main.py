from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load the dataset
df = pd.read_csv('imdb_top_1000.csv')

# Drop the 'Poster_Link', 'Certificate', 'Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4' columns
df = df.drop(['Poster_Link', 'Certificate', 'Overview', 'Runtime', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'], axis=1)

# Clean data
df = df[['Series_Title', 'Released_Year', 'Genre', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']]
df = pd.get_dummies(df, columns=['Genre'])
df = df.select_dtypes(include=[float, int]) # Select only numeric columns
df = df.select_dtypes(include=[float, int])
df = df.fillna(df.mean())

# Split dataset into training and testing sets using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(df):
    X_train, X_test = df.iloc[train_index, :-1], df.iloc[test_index, :-1]
    y_train, y_test = df.iloc[train_index, -1], df.iloc[test_index, -1]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    print(f"RMSE: {rmse}")

# Prepare new movie data
new_movie = pd.DataFrame({
    'Series_Title': ['My Movie'],
    'Released_Year': [2023],
    'Genre_Action': [0],
    'Genre_Drama': [1],
    'Genre_Horror': [0],
    'Genre_Romance': [0],
    'Genre_Sci-Fi': [0],
    'IMDB_Rating': [7.5],
    'Meta_score': [80],
    'No_of_Votes': [10000],
    'Gross': [30000000]
})

# Make prediction
new_movie = new_movie[X_train.columns]  # subset columns to match training data
success_prediction = rf.predict(new_movie)
print(success_prediction)

import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Scatter plot of predicted vs actual values')
plt.show()


# Train model and make predictions (assuming rf is already defined)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Create residual plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# Plot actual Gross vs IMDB rating
plt.scatter(X_test['IMDB_Rating'], y_test, color='blue', label='Actual Gross')

# Plot predicted Gross vs IMDB rating
plt.scatter(X_test['IMDB_Rating'], y_pred, color='red', label='Predicted Gross')

# Add axis labels and legend
plt.xlabel('IMDB Rating')
plt.ylabel('Gross')
plt.legend()

# Show the plot
plt.show()











