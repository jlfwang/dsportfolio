Predicting the selling price of a residential property depends on a number of factors, including the property age, availability of local amenities, and location.

In this challenge, you will use a dataset of real estate sales transactions to predict the price-per-unit of a property based on its features. The price-per-unit in this data is based on a unit measurement of 3.3 square meters.

*(This is an exercise from [a course on basic machine learning](https://github.com/MicrosoftDocs/ml-basics) sponsored by Microsoft)*


> **Citation**: The data used in this exercise originates from the following study:
>
> *Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with comparative approach through case-based reasoning. Applied Soft Computing, 65, 260-271.*
>
> It was obtained from the UCI dataset repository (Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science).

## Review the data

Run the following cell to load the data and view the first few rows.


```python
import pandas as pd

# load the training dataset
data = pd.read_csv('data/real_estate.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_date</th>
      <th>house_age</th>
      <th>transit_distance</th>
      <th>local_convenience_stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_per_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012.917</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012.917</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013.583</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013.500</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012.833</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
    </tr>
  </tbody>
</table>
</div>



The data consists of the following variables:

- **transaction_date** - the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
- **house_age** - the house age (in years)
- **transit_distance** - the distance to the nearest light rail station (in meters)
- **local_convenience_stores** - the number of convenience stores within walking distance
- **latitude** - the geographic coordinate, latitude
- **longitude** - the geographic coordinate, longitude
- **price_per_unit** house price of unit area (3.3 square meters)

## Train a Regression Model

Your challenge is to explore and prepare the data, identify predictive features that will help predict the **price_per_unit** label, and train a regression model that achieves the lowest Root Mean Square Error (RMSE) you can achieve (which must be less than **7**) when evaluated against a test subset of data.

Add markdown and code cells as required to create your solution.

> **Note**: There is no single "correct" solution. A sample solution is provided in [02 - Real Estate Regression Solution.ipynb](02%20-%20Real%20Estate%20Regression%20Solution.ipynb).


```python
# Your code to explore data and train a regression model

# Separating numeric features from categorical features
numeric_features = ['house_age', 'transit_distance', 'latitude', 'longitude']
categorical_features = ['local_convenience_stores']
# I am abandoning the transaction_date feature as normally the prediction of price would happen
# prior to the completion of the transaction and therefore the information wouldn't be available 
# at the time of prediction
data[numeric_features + ['price_per_unit']].describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>house_age</th>
      <th>transit_distance</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_per_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.712560</td>
      <td>1083.885689</td>
      <td>24.969030</td>
      <td>121.533361</td>
      <td>37.980193</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.392485</td>
      <td>1262.109595</td>
      <td>0.012410</td>
      <td>0.015347</td>
      <td>13.606488</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>23.382840</td>
      <td>24.932070</td>
      <td>121.473530</td>
      <td>7.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.025000</td>
      <td>289.324800</td>
      <td>24.963000</td>
      <td>121.528085</td>
      <td>27.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.100000</td>
      <td>492.231300</td>
      <td>24.971100</td>
      <td>121.538630</td>
      <td>38.450000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>28.150000</td>
      <td>1454.279000</td>
      <td>24.977455</td>
      <td>121.543305</td>
      <td>46.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>43.800000</td>
      <td>6488.021000</td>
      <td>25.014590</td>
      <td>121.566270</td>
      <td>117.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the distribution of the label
import pandas as pd
import matplotlib.pyplot as plt

# This ensures plots are displayed inline in the Jupyter notebook
%matplotlib inline

# Get the label column
label = data['price_per_unit']


# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Plot the histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('price_per_unit')

# Add a title to the Figure
fig.suptitle('Price_per_unit Distribution')

# Show the figure
fig.show()

```

    C:\Users\14389\AppData\Local\Temp/ipykernel_8760/692231007.py:31: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_4_1.png)
    



```python
# Visualize and explore the distribution of numerical features

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_5_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_5_1.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_5_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_5_3.png)
    



```python
# Visualize and explore the characteristics of categorical features
import numpy as np

# plot a bar plot for each categorical feature count

for col in categorical_features:
    counts = data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=0)   
plt.show()

```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_6_0.png)
    



```python
# Explore the correlation between the numeric features and the label

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = data[col]
    label = data['price_per_unit']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    m, b = np.polyfit(feature, label, 1)
    plt.plot(feature, m*feature + b, color = 'cyan')
    plt.xlabel(col)
    plt.ylabel('Price per Unit')
    ax.set_title('Price per Unit vs ' + col + '- correlation: ' + str(correlation))
plt.show()

```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_7_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_7_1.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_7_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_7_3.png)
    


The correlation between house_age and price is quite weak, but there seems to be a vague trend showing that higher prices tend to conincide with higher longitude and altitude and closer distance to transit.


```python
# Use boxplot to explore the relationship between the label and categorical feature
for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    data.boxplot(column = 'price_per_unit', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Price per Unit")
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_9_0.png)
    


More convinient stores around seems to indicate a potential for higher prices. (I wonder if I should get rid of the outliers in the label to make the difference more apparent...)


```python
lower_d = data.price_per_unit.quantile(0.01)
upper_d = data.price_per_unit.quantile(0.99)
print("Price first and last quantile: ", lower_d, upper_d)
df_sample = data[(data.price_per_unit>lower_d) & (data.price_per_unit<upper_d)]

for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df_sample.boxplot(column = 'price_per_unit', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Price per Unit")
plt.show()

# The difference is indeed more apparent. Price for house with 9-10 stores nearby
# almost doubles the price for house with no stores nearby
```

    Price first and last quantile:  12.8 70.88300000000001
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-real-estate-regression-challenge_files/2022-03-25-real-estate-regression-challenge_11_1.png)
    



```python
# Separating features and labels for training
# Separate features and labels
X, y = df_sample[['transit_distance', 'latitude', 'longitude', 'local_convenience_stores']].values, df_sample['price_per_unit'].values
```


```python
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))
```

    Training Set: 282 rows
    Test Set: 121 rows
    


```python
# Train the model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Fit a linear regression model on the training set
model = GradientBoostingRegressor().fit(X_train, y_train)
print (model)
```

    GradientBoostingRegressor()
    


```python
# Evaluate the Trained Model
import numpy as np

predictions = model.predict(X_test)
np.set_printoptions(suppress=True)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# RMSE too large. I'm removing the house_age feature that has little correlation with the price

```

    RMSE: 5.536161923837592
    


```python
# Improve performance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

alg = GradientBoostingRegressor()
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

model=gridsearch.best_estimator_
print(model, "\n")

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
#print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Man, getting rid of the outliers really makes a difference >0<

```

    Best parameter combination: {'learning_rate': 0.1, 'n_estimators': 50} 
    
    GradientBoostingRegressor(n_estimators=50) 
    
    RMSE: 5.624260226513916
    

## Use the Trained Model

Save your trained model, and then use it to predict the price-per-unit for the following real estate transactions:

| transaction_date | house_age | transit_distance | local_convenience_stores | latitude | longitude |
| ---------------- | --------- | ---------------- | ------------------------ | -------- | --------- |
|2013.167|16.2|289.3248|5|24.98203|121.54348|
|2013.000|13.6|4082.015|0|24.94155|121.50381|


```python
import joblib

# Save the model as a pickle file
filename = './real-estate-regression.pkl'
joblib.dump(model, filename)
```




    ['./real-estate-regression.pkl']




```python
# Your code to use the trained model

# Load the model from the file
loaded_model = joblib.load(filename)
# only taking in four features transit_distance, local_convienicen_stores, latitude, longitude
X_new = np.array([[289.3248, 5, 24.98203, 121.54348], [4082.015, 0, 24.94155, 121.50381]]).astype('float64')
print ('New sample: {}'.format(X_new))

# Use the model to predict tomorrow's rentals
results = loaded_model.predict(X_new)
for prediction in results:
    print(np.round(prediction))

# Create a numpy array containing a new observation (for example tomorrow's seasonal and weather forecast information)

```

    New sample: [[ 289.3248     5.        24.98203  121.54348]
     [4082.015      0.        24.94155  121.50381]]
    34.0
    24.0
    
