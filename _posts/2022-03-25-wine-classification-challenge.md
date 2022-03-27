---
layout: post
title: Wine Classification Challenge
tags: python, blahhhhh
excerpt_separator: <!--more-->
---

*(This is an exercise from [a course on basic machine learning](https://github.com/MicrosoftDocs/ml-basics) by Microsoft)*

Wine experts can identify wines from specific vineyards through smell and taste, but the factors that give different wines their individual charateristics are actually based on their chemical composition.

In this challenge, I will train a classification model to analyze the chemical and visual features of wine samples and classify them based on their cultivar (grape variety).

<!--more-->

> **Citation**: The data used in this exercise was originally collected by Forina, M. et al.
>
> PARVUS - An Extendible Package for Data Exploration, Classification and Correlation.
Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno,
16147 Genoa, Italy.
>
> It can be downloaded from the UCI dataset repository (Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository]([http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science). 

## Explore the data

Run the following cell to load a CSV file of wine data, which consists of 12 numeric features and a classification label with the following classes:

- **0** (*variety A*)
- **1** (*variety B*)
- **2** (*variety C*)


```python
import pandas as pd

# load the training dataset
data = pd.read_csv('data/wine.csv')
data.head()

wine_classes = ['variety A', 'variety B', 'variety C']
```

Your challenge is to explore the data and train a classification model that achieves an overall *Recall* metric of over 0.95 (95%).

> **Note**: There is no single "correct" solution. A sample solution is provided in [03 - Wine Classification Solution.ipynb](03%20-%20Wine%20Classification%20Solution.ipynb).

## Train and evaluate a model

Add markdown and code cells as required to to explore the data, train a model, and evaluate the model's predictive performance.


```python
# Your code to evaluate data, and train and evaluate a classification model
```


```python
# Separate features and labels
features = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_315_of_diluted_wines', 'Proline']
label = 'WineVariety'
X, y = data[features].values, data[label].values

for n in range(0,4):
    print("Wine", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])
```

    Wine 1 
      Features: [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0] 
      Label: 0
    Wine 2 
      Features: [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0] 
      Label: 0
    Wine 3 
      Features: [13.16, 2.36, 2.67, 18.6, 101.0, 2.8, 3.24, 0.3, 2.81, 5.68, 1.03, 3.17, 1185.0] 
      Label: 0
    Wine 4 
      Features: [14.37, 1.95, 2.5, 16.8, 113.0, 3.85, 3.49, 0.24, 2.18, 7.8, 0.86, 3.45, 1480.0] 
      Label: 0
    


```python
# Compare the feature distributions for each label value
from matplotlib import pyplot as plt
%matplotlib inline

for col in features:
    data.boxplot(column=col, by='WineVariety', figsize=(6,6))
    plt.title(col)
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_1.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_3.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_4.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_5.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_6.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_7.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_8.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_9.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_10.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_11.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_7_12.png)
    



```python
# Split into train test groups
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
```

    Training cases: 124
    Test cases: 54
    


```python
# Experiment with the Logistic Regression

# Train the model
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(X_train, y_train)
print (model)
```

    LogisticRegression(C=100.0, max_iter=10000)
    


```python
predictions = model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual labels:    ' ,y_test)
```

    Predicted labels:  [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 2 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0 2
     1 1 2 0 0 1 1 1 0 2 1 2 0 2 2 0 2]
    Actual labels:     [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0 2
     1 1 2 0 0 1 1 1 0 2 1 2 0 2 2 0 2]
    


```python
from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(y_test, predictions))

from sklearn. metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.metrics import precision_score, recall_score

print("Overall Precision:", precision_score(y_test, predictions, average='weighted'))
print("Overall Recall:", recall_score(y_test, predictions, average='weighted'))

```

    Accuracy:  0.9814814814814815
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        19
               1       1.00      0.95      0.98        22
               2       0.93      1.00      0.96        13
    
        accuracy                           0.98        54
       macro avg       0.98      0.98      0.98        54
    weighted avg       0.98      0.98      0.98        54
    
    Overall Precision: 0.9828042328042328
    Overall Recall: 0.9814814814814815
    


```python
# More eval through confusion matrix
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
mcm = confusion_matrix(y_test, predictions)
print(mcm)
```

    [[19  0  0]
     [ 0 21  1]
     [ 0  0 13]]
    


```python
# Visualize the confusion matrix
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(wine_classes))
plt.xticks(tick_marks, wine_classes, rotation=45)
plt.yticks(tick_marks, wine_classes)
plt.xlabel("Predicted type")
plt.ylabel("Actual type")
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-wine-classification-challenge_files/2022-03-25-wine-classification-challenge_13_0.png)
    



```python
# Save the model for future use

import joblib

# Save the model as a pickle file
filename = './wine_model.pkl'
joblib.dump(model, filename)
```




    ['./wine_model.pkl']



## Use the model with new data observation

When you're happy with your model's predictive performance, save it and then use it to predict classes for the following two new wine samples:

- \[13.72,1.43,2.5,16.7,108,3.4,3.67,0.19,2.04,6.8,0.89,2.87,1285\]
- \[12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520\]



```python
# Your code to predict classes for the two new samples
# Load the model from the file
model = joblib.load(filename)

# The model accepts an array of feature arrays (so you can predict the classes of multiple penguin observations in a single call)
# We'll create an array with a single array of features, representing one penguin
x_new = np.array([[13.72,1.43,2.5,16.7,108,3.4,3.67,0.19,2.04,6.8,0.89,2.87,1285], [12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520]])
print ('New sample: {}'.format(x_new))

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one penguin, so our prediction is the first one in the resulting array.
pred1 = model.predict(x_new)[0]
pred2 = model.predict(x_new)[1]
print('First predicted class is', wine_classes[pred1])
print('Second predicted class is', wine_classes[pred2])
```

    New sample: [[1.372e+01 1.430e+00 2.500e+00 1.670e+01 1.080e+02 3.400e+00 3.670e+00
      1.900e-01 2.040e+00 6.800e+00 8.900e-01 2.870e+00 1.285e+03]
     [1.237e+01 9.400e-01 1.360e+00 1.060e+01 8.800e+01 1.980e+00 5.700e-01
      2.800e-01 4.200e-01 1.950e+00 1.050e+00 1.820e+00 5.200e+02]]
    First predicted class is variety A
    Second predicted class is variety B
    
