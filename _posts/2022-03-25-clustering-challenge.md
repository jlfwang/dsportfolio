Clustering is an *unsupervised* machine learning technique in which you train a model to group similar entities into clusters based on their features.

In this exercise, you must separate a dataset consisting of three numeric features (**A**, **B**, and **C**) into clusters. Run the cell below to load the data.

*(This is an exercise from [a course on basic machine learning](https://github.com/MicrosoftDocs/ml-basics) sponsored by Microsoft)*


```python
import pandas as pd

data = pd.read_csv('data/clusters.csv')
data.sample(10)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>1.049570</td>
      <td>2.053523</td>
      <td>2.597098</td>
    </tr>
    <tr>
      <th>796</th>
      <td>-0.206279</td>
      <td>-0.096671</td>
      <td>0.378876</td>
    </tr>
    <tr>
      <th>391</th>
      <td>3.510686</td>
      <td>3.873458</td>
      <td>1.789713</td>
    </tr>
    <tr>
      <th>667</th>
      <td>0.208767</td>
      <td>-0.410377</td>
      <td>0.405790</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2.124788</td>
      <td>3.636222</td>
      <td>2.800396</td>
    </tr>
    <tr>
      <th>563</th>
      <td>0.501970</td>
      <td>0.669061</td>
      <td>1.192833</td>
    </tr>
    <tr>
      <th>422</th>
      <td>-1.354736</td>
      <td>0.458732</td>
      <td>-0.425959</td>
    </tr>
    <tr>
      <th>868</th>
      <td>2.485840</td>
      <td>2.568866</td>
      <td>1.286884</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2.008746</td>
      <td>0.467115</td>
      <td>0.851036</td>
    </tr>
    <tr>
      <th>326</th>
      <td>-0.298586</td>
      <td>0.129519</td>
      <td>0.274402</td>
    </tr>
  </tbody>
</table>
</div>



Your challenge is to identify the number of discrete clusters present in the data, and create a clustering model that separates the data into that number of clusters. You should also visualize the clusters to evaluate the level of separation achieved by your model.

Add markdown and code cells as required to create your solution.

> **Note**: There is no single "correct" solution. A sample solution is provided in [04 - Clustering Solution.ipynb](04%20-%20Clustering%20Solution.ipynb).


```python
# Your code to create a clustering solution
```


```python
features = data[data.columns[0:3]]
features.sample(10)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Normalize the numeric features so they're on the same scale
# scaled_features = MinMaxScaler().fit_transform(features[data.columns[0:2]])

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)


```


```python
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(features_2d[:,0],features_2d[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-clustering-challenge_files/2022-03-25-clustering-challenge_5_0.png)
    



```python
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

# Create 10 models with 1 to 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    # Fit the data points
    kmeans.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmeans.inertia_)
    
#Plot the WCSS values onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-clustering-challenge_files/2022-03-25-clustering-challenge_6_0.png)
    


# K-Means Clustering


```python
from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters
```




    array([2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 1, 0, 2, 0, 1, 0, 1, 0, 0, 0, 2, 1,
           2, 2, 1, 2, 1, 2, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 0, 0, 0, 1, 0,
           1, 2, 2, 1, 1, 0, 2, 0, 1, 1, 1, 1, 0, 2, 1, 2, 2, 2, 2, 2, 0, 1,
           2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 2, 0,
           0, 0, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 0, 2, 1, 0, 0, 1,
           0, 0, 2, 2, 2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0,
           0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 2, 2, 0,
           2, 0, 2, 0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 0, 1, 0, 2,
           0, 0, 0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0,
           2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 0, 2, 1, 0, 1, 1, 2, 1,
           1, 2, 2, 1, 0, 1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 0, 1, 0,
           1, 0, 1, 2, 1, 2, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 1, 2, 0, 1, 1, 0,
           2, 0, 0, 2, 0, 0, 1, 2, 1, 2, 2, 2, 1, 0, 0, 2, 1, 1, 2, 2, 2, 1,
           1, 0, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 0, 2, 1, 0, 1, 2, 2, 0, 2, 0,
           2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 2, 1, 2, 2, 0, 1,
           1, 2, 2, 1, 1, 0, 0, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 2, 0, 2, 2, 0,
           1, 1, 1, 1, 2, 1, 1, 2, 0, 0, 2, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2,
           1, 2, 0, 1, 2, 1, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 1,
           1, 0, 2, 1, 2, 1, 2, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 1, 0, 1, 1, 0,
           1, 2, 2, 2, 2, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 1, 2, 2, 1, 2,
           1, 1, 2, 1, 0, 0, 1, 1, 2, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 1, 2,
           0, 2, 2, 0, 1, 1, 0, 2, 0, 1, 1, 2, 2, 2, 0, 0, 2, 2, 2, 2, 1, 2,
           2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 0, 0, 0, 2, 0,
           2, 2, 1, 0, 0, 2, 2, 1, 0, 1, 0, 0, 1, 2, 1, 1, 2, 2, 1, 0, 1, 1,
           0, 0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2,
           0, 1, 2, 1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 0, 0,
           0, 1, 0, 0, 0, 0, 2, 1, 0, 2, 0, 0, 2, 0, 1, 2, 1, 0, 1, 0, 2, 0,
           0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2, 2,
           2, 2, 0, 0, 2, 2, 1, 0, 1, 1, 1, 2, 2, 1, 2, 0, 0, 2, 2, 0, 1, 0,
           0, 2, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 2, 1, 0, 2, 1, 1, 1, 1, 2, 1,
           0, 2, 1, 2, 2, 2, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 0, 1, 1, 1, 1, 1,
           1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 2, 2, 0, 2, 2, 1,
           1, 1, 0, 1, 2, 1, 2, 2, 0, 2, 2, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 2,
           0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 2, 2, 2, 2,
           2, 2, 1, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 0, 0,
           2, 1, 2, 1, 2, 1, 1, 2, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1,
           2, 1, 2, 1, 2, 2, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0,
           0, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 1, 2, 0, 2, 1, 2,
           1, 0, 0, 0, 0, 1, 1, 2, 0, 2, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
           0, 0, 1, 1, 2, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 2, 0, 0, 1, 2, 0, 1,
           2, 1, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 2, 2, 0, 0,
           1, 2, 2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 1, 2, 0, 2, 0, 2, 1, 0, 2, 1,
           0, 1, 0, 1, 1, 2, 1, 1, 1, 0, 2, 1, 0, 2, 1, 2, 0, 2, 2, 2, 0, 2,
           2, 1, 1, 2, 0, 1, 0, 2, 1, 1, 2, 1, 0, 0, 2, 2, 0, 2, 0, 2, 0, 0,
           1, 0, 2, 0, 1, 0, 1, 2, 1, 1, 2, 0, 0, 1, 0, 1, 0, 1, 2, 0, 2, 1,
           2, 2, 0, 2, 1, 0, 1, 0, 2, 1])




```python
import matplotlib.pyplot as plt
%matplotlib inline

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-clustering-challenge_files/2022-03-25-clustering-challenge_9_0.png)
    


# Hierarchical Clustering


```python
from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters
```




    array([1, 1, 0, 2, 1, 1, 2, 0, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0,
           1, 1, 0, 1, 2, 1, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0,
           2, 1, 1, 2, 2, 0, 1, 0, 0, 2, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,
           0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2,
           0, 0, 1, 1, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 2, 0, 2, 0, 0, 2, 1, 1, 0,
           1, 0, 1, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 1, 0, 0, 2, 0, 1,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0,
           1, 2, 0, 2, 2, 2, 1, 1, 0, 0, 2, 1, 2, 0, 0, 1, 2, 0, 0, 2, 1, 2,
           0, 1, 1, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0,
           2, 0, 0, 1, 2, 1, 1, 0, 2, 0, 0, 1, 2, 2, 2, 1, 2, 1, 0, 2, 0, 0,
           1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 2, 0, 0, 1, 2, 2, 0, 1, 1, 2,
           2, 0, 2, 1, 1, 0, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0,
           0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 0, 0,
           0, 1, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 0, 1, 0,
           0, 2, 2, 2, 1, 0, 2, 0, 0, 0, 1, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1,
           2, 1, 0, 2, 1, 2, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 2,
           0, 0, 0, 2, 1, 2, 0, 1, 2, 1, 0, 0, 1, 1, 2, 0, 0, 2, 0, 2, 0, 0,
           0, 1, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 2, 1,
           2, 2, 1, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 2, 0, 2, 0, 0, 0, 2, 1,
           0, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 2, 1,
           1, 2, 1, 2, 1, 0, 1, 2, 0, 1, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 1, 0,
           0, 1, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 1, 2, 2, 1, 1, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 1,
           0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 2, 0, 1, 0,
           0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0,
           1, 1, 0, 0, 1, 1, 2, 0, 0, 0, 2, 1, 1, 2, 1, 0, 0, 1, 1, 0, 2, 0,
           0, 1, 2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 2, 0, 2, 1, 0,
           0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2, 1, 0, 2, 2, 0, 2, 2,
           2, 0, 2, 2, 0, 1, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 1, 1, 0,
           2, 0, 0, 2, 1, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 2, 1, 0, 1, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 0,
           1, 0, 1, 2, 0, 2, 2, 1, 2, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 2,
           1, 2, 1, 2, 1, 1, 0, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 1, 0, 1, 2, 0,
           0, 0, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0, 2, 1,
           2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
           0, 0, 2, 0, 1, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2,
           1, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0,
           2, 1, 1, 0, 0, 0, 1, 1, 2, 0, 1, 1, 2, 1, 0, 0, 0, 1, 0, 0, 1, 2,
           0, 2, 0, 2, 2, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 1, 0, 1, 0, 0, 0, 1,
           1, 2, 2, 1, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
           0, 0, 1, 0, 2, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 0, 2, 1, 0, 1, 2,
           0, 1, 0, 0, 2, 0, 2, 0, 1, 0], dtype=int64)




```python
import matplotlib.pyplot as plt

%matplotlib inline

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, agg_clusters)

```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-clustering-challenge_files/2022-03-25-clustering-challenge_12_0.png)
    

