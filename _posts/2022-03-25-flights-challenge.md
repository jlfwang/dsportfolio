In this challenge, you'll explore a real-world dataset containing flights data from the US Department of Transportation.

*(This is an exercise from [a course on basic machine learning](https://github.com/MicrosoftDocs/ml-basics) sponsored by Microsoft)*


Let's start by loading and viewing the data.


```python
import pandas as pd

df_flights = pd.read_csv('data/flights.csv')

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
</div>



The dataset contains observations of US domestic flights in 2013, and consists of the following fields:

- **Year**: The year of the flight (all records are from 2013)
- **Month**: The month of the flight
- **DayofMonth**: The day of the month on which the flight departed
- **DayOfWeek**: The day of the week on which the flight departed - from 1 (Monday) to 7 (Sunday)
- **Carrier**: The two-letter abbreviation for the airline.
- **OriginAirportID**: A unique numeric identifier for the departure aiport
- **OriginAirportName**: The full name of the departure airport
- **OriginCity**: The departure airport city
- **OriginState**: The departure airport state
- **DestAirportID**: A unique numeric identifier for the destination aiport
- **DestAirportName**: The full name of the destination airport
- **DestCity**: The destination airport city
- **DestState**: The destination airport state
- **CRSDepTime**: The scheduled departure time
- **DepDelay**: The number of minutes departure was delayed (flight that left ahead of schedule have a negative value)
- **DelDelay15**: A binary indicator that departure was delayed by more than 15 minutes (and therefore considered "late")
- **CRSArrTime**: The scheduled arrival time
- **ArrDelay**: The number of minutes arrival was delayed (flight that arrived ahead of schedule have a negative value)
- **ArrDelay15**: A binary indicator that arrival was delayed by more than 15 minutes (and therefore considered "late")
- **Cancelled**: A binary indicator that the flight was cancelled

Your challenge is to explore the flight data to analyze possible factors that affect delays in departure or arrival of a flight.

1. Start by cleaning the data.
    - Identify any null or missing data, and impute appropriate replacement values.
    - Identify and eliminate any outliers in the **DepDelay** and **ArrDelay** columns.
2. Explore the cleaned data.
    - View summary statistics for the numeric fields in the dataset.
    - Determine the distribution of the **DepDelay** and **ArrDelay** columns.
    - Use statistics, aggregate functions, and visualizations to answer the following questions:
        - *What are the average (mean) departure and arrival delays?*
        - *How do the carriers compare in terms of arrival delay performance?*
        - *Is there a noticable difference in arrival delays for different days of the week?*
        - *Which departure airport has the highest average departure delay?*
        - *Do **late** departures tend to result in longer arrival delays than on-time departures?*
        - *Which route (from origin airport to destination airport) has the most **late** arrivals?*
        - *Which route has the highest average arrival delay?*
        
Add markdown and code cells as required to create your solution.

> **Note**: There is no single "correct" solution. A sample solution is provided in [01 - Flights Challenge.ipynb](01%20-%20Flights%20Solution.ipynb).


```python
# Your code to explore the data

# Identify any null or missing data, and impute appropriate replacement values.
df_flights.isnull().sum()

```




    Year                    0
    Month                   0
    DayofMonth              0
    DayOfWeek               0
    Carrier                 0
    OriginAirportID         0
    OriginAirportName       0
    OriginCity              0
    OriginState             0
    DestAirportID           0
    DestAirportName         0
    DestCity                0
    DestState               0
    CRSDepTime              0
    DepDelay                0
    DepDel15             2761
    CRSArrTime              0
    ArrDelay                0
    ArrDel15                0
    Cancelled               0
    dtype: int64



All missing values are in the DepDel15 column, which could be inferred from values in column DepDelay 


```python
# Impute the DepDel15 column based on DepDelay values
df_flights['DepDel15'] = df_flights['DepDel15'].fillna(
    lambda row: df_flights['DepDelay'] > 15,
)
print(df_flights.isnull().sum())
```

    Year                 0
    Month                0
    DayofMonth           0
    DayOfWeek            0
    Carrier              0
    OriginAirportID      0
    OriginAirportName    0
    OriginCity           0
    OriginState          0
    DestAirportID        0
    DestAirportName      0
    DestCity             0
    DestState            0
    CRSDepTime           0
    DepDelay             0
    DepDel15             0
    CRSArrTime           0
    ArrDelay             0
    ArrDel15             0
    Cancelled            0
    dtype: int64
    

Identify and eliminate any outliers in the **DepDelay** and **ArrDelay** columns.


```python
from matplotlib import pyplot as plt

# A function to visualize the distribution of data
def show_distribution(var_data):
    '''
    This function will make a distribution (graph) and display it
    '''
    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]
    rng = max_val - min_val
    var = var_data.var()
    std = var_data.std()
    
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\nRange:{:.2f}\nVariance:{:.2f}\nStd.Dev:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val, rng, var, std))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle(str(var_data.name) + " Distribution")

    # Show the figure
    fig.show()

col1 = df_flights['DepDelay']
show_distribution(col1)

col2 = df_flights['ArrDelay']
show_distribution(col2)


```

    Minimum:-63.00
    Mean:10.35
    Median:-1.00
    Mode:-3.00
    Maximum:1425.00
    Range:1488.00
    Variance:1272.61
    Std.Dev:35.67
    
    Minimum:-75.00
    Mean:6.50
    Median:-3.00
    Mode:0.00
    Maximum:1440.00
    Range:1515.00
    Variance:1461.56
    Std.Dev:38.23
    
    

    C:\Users\14389\AppData\Local\Temp/ipykernel_9768/329630434.py:46: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()
    C:\Users\14389\AppData\Local\Temp/ipykernel_9768/329630434.py:46: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_7_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_7_3.png)
    



```python
def show_density(var_data):
    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title(str(var_data.name) + 'Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of StudyHours
show_density(col1)
show_density(col2)
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_8_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_8_1.png)
    



```python
# Measure the variance

for col_name in ['DepDelay','ArrDelay']:
    col = df_flights[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))

# Drawing the lines to define outliers
print()
lower_d = df_flights.DepDelay.quantile(0.05)
upper_d = df_flights.DepDelay.quantile(0.95)
print("DepDelay first and last quantile: ", lower_d, upper_d)
print()
lower_a = df_flights.DepDelay.quantile(0.05)
upper_a = df_flights.DepDelay.quantile(0.95)
print("ArrDelay first and last quantile: ", lower_a, upper_a)
```

    
    DepDelay:
     - Range: 1488.00
     - Variance: 1272.61
     - Std.Dev: 35.67
    
    ArrDelay:
     - Range: 1515.00
     - Variance: 1461.56
     - Std.Dev: 38.23
    
    DepDelay first and last quantile:  -8.0 70.0
    
    ArrDelay first and last quantile:  -8.0 70.0
    


```python
# Based on the above information, it seems reasonable to rid of the observations 
# below the 0.01th percentile and the value above which 99% of the data reside.
col1 = df_flights[(df_flights.DepDelay>lower_d) & (df_flights.DepDelay<upper_d)]['DepDelay']
col2 = df_flights[(df_flights.ArrDelay>lower_a) & (df_flights.DepDelay<upper_a)]['ArrDelay']

# Call the function
show_distribution(col1)
show_distribution(col2)
```

    Minimum:-7.00
    Mean:5.09
    Median:-1.00
    Mode:-3.00
    Maximum:69.00
    Range:76.00
    Variance:212.58
    Std.Dev:14.58
    
    Minimum:-7.00
    Mean:9.67
    Median:3.00
    Mode:0.00
    Maximum:204.00
    Range:211.00
    Variance:322.68
    Std.Dev:17.96
    
    

    C:\Users\14389\AppData\Local\Temp/ipykernel_9768/329630434.py:46: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()
    C:\Users\14389\AppData\Local\Temp/ipykernel_9768/329630434.py:46: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_10_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_10_3.png)
    



```python
# Density of the cleand DepDelay and ArrDelay data
show_density(col1)
show_density(col2)
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_11_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_11_1.png)
    


Both columns appear right skewed, so let's try normalize them.


```python
df_sample = df_flights[(df_flights.DepDelay>lower_d) & (df_flights.DepDelay<upper_d) & (df_flights.ArrDelay>lower_a) & (df_flights.ArrDelay<upper_a)]
```

**Summary statistics for the numeric fields.**

Numeric fields consists of only the DepDelay and ArrDelay columns 
<br>
Year, Month, DayofMonth, DayOfWeek, OriginAirportID, DestAirportID, CRSDepTime, CRSArrTime are all categorical

Answer the following questions:
- What are the average (mean) departure and arrival delays?
    - average departure delay: 8.22 min
    - average arrival delay:11.95 min
- How do the carriers compare in terms of arrival delay performance?
- Is there a noticable difference in arrival delays for different days of the week?
- Which departure airport has the highest average departure delay?
- Do late departures tend to result in longer arrival delays than on-time departures?
- Which route (from origin airport to destination airport) has the most late arrivals?
- Which route has the highest average arrival delay?


```python
# How do the carriers compare in terms of arrival delay performance?
df_sample.boxplot(column='ArrDelay', by='Carrier', figsize=(10,5))
```




    <AxesSubplot:title={'center':'ArrDelay'}, xlabel='Carrier'>




    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_16_1.png)
    



```python
df_sample.groupby('Carrier').ArrDelay.mean().sort_values()

# The follow result shows that HA airline has the smallest arrival delay of an average 2.66 min
# While MQ airline has the longest arrival delay of 11.68 min

# To find out more details about how each airline is performing, uncomment the following line
#df_sample.groupby('Carrier').ArrDelay.describe()

```




    Carrier
    HA     2.663559
    AS     6.727634
    DL     7.170074
    YV     7.964551
    OO     8.172333
    VX     8.177463
    US     8.525956
    9E     8.606802
    WN     8.858771
    FL     9.303314
    F9     9.631191
    UA     9.829855
    AA     9.859843
    EV    11.160135
    B6    11.457301
    MQ    11.684145
    Name: ArrDelay, dtype: float64




```python
# Is there a noticable difference in arrival delays for different days of the week?
print(df_sample.groupby('DayOfWeek').ArrDelay.mean().sort_values())
shortest_delay = df_sample.groupby('DayOfWeek').ArrDelay.mean().min()
longest_dellay = df_sample.groupby('DayOfWeek').ArrDelay.mean().max()
r = longest_dellay - shortest_delay
print('\n Range of Delay: {:.2f}'.format(r))

```

    DayOfWeek
    6     7.733801
    2     8.013798
    3     8.807281
    7     8.903637
    1     9.101978
    5     9.756578
    4    10.145735
    Name: ArrDelay, dtype: float64
    
     Range of Delay: 2.41
    


```python
# Let's visualize the data
df_sample.boxplot(column='ArrDelay', by='DayOfWeek', figsize=(10,5))
```




    <AxesSubplot:title={'center':'ArrDelay'}, xlabel='DayOfWeek'>




    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_19_1.png)
    


From the above results, we could see that there are some differences in arrival delays for different days of the week, but the difference is quite small, with a range of 2.41 min


```python
# Which departure airport has the highest average departure delay?

df_sample.groupby('OriginAirportName').ArrDelay.mean().sort_values(ascending = False)

# LaGuardia has the highest average departure delay of 12.56 min
```




    OriginAirportName
    LaGuardia                        12.564659
    John F. Kennedy International    12.507686
    Newark Liberty International     12.223352
    Chicago Midway International     11.640951
    Philadelphia International       11.163102
                                       ...    
    Sacramento International          5.471455
    Long Beach Airport                5.334158
    Bob Hope                          5.206651
    Kahului Airport                   4.943396
    Honolulu International            4.578748
    Name: ArrDelay, Length: 70, dtype: float64




```python
# Do late departures tend to result in longer arrival delays than on-time departures?

df_sample.boxplot(column='ArrDelay', by='DepDel15', figsize=(10,5))

```




    <AxesSubplot:title={'center':'ArrDelay'}, xlabel='DepDel15'>




    
![png]({{ site.baseurl }}/assets/img/2022-03-25-flights-challenge_files/2022-03-25-flights-challenge_22_1.png)
    



```python
df_sample.groupby('DepDel15').ArrDelay.mean()

# Seems like a delay in departure does result in longer arrival delays
```




    DepDel15
    0.0     3.127341
    1.0    26.666972
    Name: ArrDelay, dtype: float64




```python
# Which route (from origin airport to destination airport) has the most late arrivals? 
# Requires a new feature that describes route

df_sample['Route'] = df_sample['OriginAirportName'] + ' - ' + df_sample['DestAirportName']
df_sample.head()
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
</div>




```python
df_sample.groupby('Route').ArrDel15.count().sort_values(ascending = False)

# SF to LA has the most late arrivals
```




    Route
    San Francisco International - Los Angeles International               586
    Kahului Airport - Honolulu International                              549
    Los Angeles International - San Francisco International               523
    Honolulu International - Kahului Airport                              522
    Los Angeles International - McCarran International                    459
                                                                         ... 
    Cleveland-Hopkins International - Austin - Bergstrom International      1
    Lambert-St. Louis International - Washington Dulles International       1
    Ronald Reagan Washington National - Luis Munoz Marin International      1
    Southwest Florida International - Theodore Francis Green State          1
    Washington Dulles International - Lambert-St. Louis International       1
    Name: ArrDel15, Length: 2464, dtype: int64




```python
df_sample.groupby('Route').ArrDelay.mean().sort_values(ascending = False)

# Pittsburgh International to Raleigh-Durham International has the highest average arrival delay

```




    Route
    Pittsburgh International - Raleigh-Durham International               63.0
    Ronald Reagan Washington National - Luis Munoz Marin International    54.0
    Southwest Florida International - Theodore Francis Green State        44.0
    Minneapolis-St Paul International - Jacksonville International        40.5
    Newark Liberty International - Will Rogers World                      37.0
                                                                          ... 
    Eppley Airfield - Orlando International                               -6.0
    William P Hobby - Philadelphia International                          -7.0
    Salt Lake City International - Washington Dulles International        -7.0
    Lambert-St. Louis International - Washington Dulles International     -7.0
    Austin - Bergstrom International - Cleveland-Hopkins International    -7.0
    Name: ArrDelay, Length: 2464, dtype: float64


