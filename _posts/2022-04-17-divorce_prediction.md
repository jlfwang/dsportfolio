---
layout: post
title: Predicting Divorce
---
This is a classification challenge to predict divorce based on answers to a series of Likert scale questions.

> **Citation**: The data used in this exercise was originally provided by csafrit on Kaggle: [Predicting Divorce](https://www.kaggle.com/datasets/csafrit2/predicting-divorce)

### About the data

The following cell loads a CSV file of the divorce questionnaire data, which consists of 54 questions with answers provided by couples on a scale of 0-4 with 0 being the lowest level of agreement with the statement and 4 being the highest. The last column 'Divorce_Y_N' indicate if divorce happened.

**Here is a complete list of the questions in the questionnaire:**

1. If one of us apologizes when our discussion deteriorates, the discussion ends.
1. I know we can ignore our differences, even if things get hard sometimes.
1. When we need it, we can take our discussions with my spouse from the beginning and correct it.
1. When I discuss with my spouse, to contact him will eventually work.
1. The time I spent with my wife is special for us.
1. We don't have time at home as partners.
1. We are like two strangers who share the same environment at home rather than family.
1. I enjoy our holidays with my wife.
1. I enjoy traveling with my wife.
1. Most of our goals are common to my spouse.
1. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
1. My spouse and I have similar values in terms of personal freedom.
1. My spouse and I have similar sense of entertainment.
1. Most of our goals for people (children, friends, etc.) are the same.
1. Our dreams with my spouse are similar and harmonious.
1. We're compatible with my spouse about what love should be.
1. We share the same views about being happy in our life with my spouse
1. My spouse and I have similar ideas about how marriage should be
1. My spouse and I have similar ideas about how roles should be in marriage
1. My spouse and I have similar values in trust.
1. I know exactly what my wife likes.
1. I know how my spouse wants to be taken care of when she/he sick.
1. I know my spouse's favorite food.
1. I can tell you what kind of stress my spouse is facing in her/his life.
1. I have knowledge of my spouse's inner world.
1. I know my spouse's basic anxieties.
1. I know what my spouse's current sources of stress are.
1. I know my spouse's hopes and wishes.
1. I know my spouse very well.
1. I know my spouse's friends and their social relationships.
1. I feel aggressive when I argue with my spouse.
1. When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .
1. I can use negative statements about my spouse's personality during our discussions.
1. I can use offensive expressions during our discussions.
1. I can insult my spouse during our discussions.
1. I can be humiliating when we discussions.
1. My discussion with my spouse is not calm.
1. I hate my spouse's way of open a subject.
1. Our discussions often occur suddenly.
1. We're just starting a discussion before I know what's going on.
1. When I talk to my spouse about something, my calm suddenly breaks.
1. When I argue with my spouse, ı only go out and I don't say a word.
1. I mostly stay silent to calm the environment a little bit.
1. Sometimes I think it's good for me to leave home for a while.
1. I'd rather stay silent than discuss with my spouse.
1. Even if I'm right in the discussion, I stay silent to hurt my spouse.
1. When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
1. I feel right in our discussions.
1. I have nothing to do with what I've been accused of.
1. I'm not actually the one who's guilty about what I'm accused of.
1. I'm not the one who's wrong about problems at home.
1. I wouldn't hesitate to tell my spouse about her/his inadequacy.
1. When I discuss, I remind my spouse of her/his inadequacy.
1. I'm not afraid to tell my spouse about her/his incompetence.

The data is quite complete and requires no cleaning.

```python
import pandas as pd

# load the dataset
!wget https://raw.githubusercontent.com/JessiDub/dsportfolio/main/divorce.csv
divorce = pd.read_csv('divorce.csv',delimiter=',',header='infer')
divorce.info()
divorce.describe()
divorce.head()
```

    --2022-04-16 20:04:24--  https://raw.githubusercontent.com/JessiDub/dsportfolio/main/divorce.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.108.133, 185.199.111.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19579 (19K) [text/plain]
    Saving to: 'divorce.csv.6'
    
         0K .......... .........                                  100% 3.92M=0.005s
    
    2022-04-16 20:04:24 (3.92 MB/s) - 'divorce.csv.6' saved [19579/19579]
    
    

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 170 entries, 0 to 169
    Data columns (total 55 columns):
     #   Column                         Non-Null Count  Dtype
    ---  ------                         --------------  -----
     0   Sorry_end                      170 non-null    int64
     1   Ignore_diff                    170 non-null    int64
     2   begin_correct                  170 non-null    int64
     3   Contact                        170 non-null    int64
     4   Special_time                   170 non-null    int64
     5   No_home_time                   170 non-null    int64
     6   2_strangers                    170 non-null    int64
     7   enjoy_holiday                  170 non-null    int64
     8   enjoy_travel                   170 non-null    int64
     9   common_goals                   170 non-null    int64
     10  harmony                        170 non-null    int64
     11  freeom_value                   170 non-null    int64
     12  entertain                      170 non-null    int64
     13  people_goals                   170 non-null    int64
     14  dreams                         170 non-null    int64
     15  love                           170 non-null    int64
     16  happy                          170 non-null    int64
     17  marriage                       170 non-null    int64
     18  roles                          170 non-null    int64
     19  trust                          170 non-null    int64
     20  likes                          170 non-null    int64
     21  care_sick                      170 non-null    int64
     22  fav_food                       170 non-null    int64
     23  stresses                       170 non-null    int64
     24  inner_world                    170 non-null    int64
     25  anxieties                      170 non-null    int64
     26  current_stress                 170 non-null    int64
     27  hopes_wishes                   170 non-null    int64
     28  know_well                      170 non-null    int64
     29  friends_social                 170 non-null    int64
     30  Aggro_argue                    170 non-null    int64
     31  Always_never                   170 non-null    int64
     32  negative_personality           170 non-null    int64
     33  offensive_expressions          170 non-null    int64
     34  insult                         170 non-null    int64
     35  humiliate                      170 non-null    int64
     36  not_calm                       170 non-null    int64
     37  hate_subjects                  170 non-null    int64
     38  sudden_discussion              170 non-null    int64
     39  idk_what's_going_on            170 non-null    int64
     40  calm_breaks                    170 non-null    int64
     41  argue_then_leave               170 non-null    int64
     42  silent_for_calm                170 non-null    int64
     43  good_to_leave_home             170 non-null    int64
     44  silence_instead_of_discussion  170 non-null    int64
     45  silence_for_harm               170 non-null    int64
     46  silence_fear_anger             170 non-null    int64
     47  I'm_right                      170 non-null    int64
     48  accusations                    170 non-null    int64
     49  I'm_not_guilty                 170 non-null    int64
     50  I'm_not_wrong                  170 non-null    int64
     51  no_hesitancy_inadequate        170 non-null    int64
     52  you're_inadequate              170 non-null    int64
     53  incompetence                   170 non-null    int64
     54  Divorce_Y_N                    170 non-null    int64
    dtypes: int64(55)
    memory usage: 73.2 KB
    

### Explore the data


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



Likert scale questionnaires tend to use techniques, such as **negative items** or questions which are similar but worded differently, to ensure its validity, so I suspect the 54 questions could be divided into groups, where the with-in group items highly correlatiove with each other. 

For example, the following questions could possibly form a group to provide information on how the couple handles their differences in opinions:
1. If one of us apologizes when our discussion deteriorates, the discussion ends.
1. I know we can ignore our differences, even if things get hard sometimes.
1. When we need it, we can take our discussions with my spouse from the beginning and correct it.
1. When I discuss with my spouse, to contact him will eventually work.

And the answers to these question would probably highly correlate with each other.


```python
import seaborn as sns; sns.set_theme()

accept_difference = divorce.iloc[:, 0:4]
sns.heatmap(accept_difference.corr(), vmin=-1, vmax=1, annot=True)

```




    <AxesSubplot:>




    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_5_1.png)
    


Therefore we could cosolidate these columns into a new feature called *accept_difference* that averages the values of these columns and provides some general information on how well the couple handles their differences in opinions. 


```python
import pandas as pd
divorce["accept_difference"] = divorce.iloc[:, 0:4].mean(axis=1)
```

Similarly, the following questions could be grouped together to reveal how much the couple's goals and interests align:

10. Most of our goals are common to my spouse.
1. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
1. My spouse and I have similar values in terms of personal freedom.
1. My spouse and I have similar sense of entertainment.
1. Most of our goals for people (children, friends, etc.) are the same.
1. Our dreams with my spouse are similar and harmonious.

And answers to these questions probably highly correlate with each other as well.


```python
alignment = divorce.iloc[:, 9:15]
sns.heatmap(alignment.corr(), vmin=-1, vmax=1, annot=True)
```




    <AxesSubplot:>




    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_9_1.png)
    


Let's consolidate these questions into another new feature named *alignment* using the average values of the columns.


```python
divorce["alignment"] = divorce.iloc[:, 9:15].mean(axis=1)
```

Same thing with 
1. questions 16-20 on how couple understand the concept of marriage, could be grouped as "marriage_concept"
2. questions 21-30 on how well the couple know each other, could be grouped as "know_each_other"
3. questions 31-37 how aggressive when the couple interact with each other, could be grouped as "aggressive" 


```python
marriage_concept = divorce.iloc[:, 15:20]
sns.heatmap(marriage_concept.corr(), vmin=-1, vmax=1, annot=True)
divorce["marriage_concept"] = divorce.iloc[:, 15:20].mean(axis=1)
```


    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_13_0.png)
    



```python
know_each_other = divorce.iloc[:, 20:30]
sns.heatmap(know_each_other.corr(), vmin=-1, vmax=1, annot=True)
divorce["know_each_other"] = divorce.iloc[:, 20:30].mean(axis=1)
```


    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_14_0.png)
    



```python
agressive = divorce.iloc[:, 30:37]
sns.heatmap(agressive.corr(), vmin=-1, vmax=1, annot=True)
divorce["agressive"] = divorce.iloc[:, 30:37].mean(axis=1)
```


    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_15_0.png)
    


I have omitted quite a few questions at these point, including:

Some questions about their time spent together

5.	The time I spent with my wife is special for us.
6.	We don't have time at home as partners.
7.	We are like two strangers who share the same environment at home rather than family.
8.	I enjoy our holidays with my wife.
9.	I enjoy traveling with my wife.

And more details on the way their converse/argue

37. I hate my spouse's way of open a subject.
39.	Our discussions often occur suddenly.
40.	We're just starting a discussion before I know what's going on.
41.	When I talk to my spouse about something, my calm suddenly breaks.
42.	When I argue with my spouse, ı only go out and I don't say a word.
43.	I mostly stay silent to calm the environment a little bit.
44.	Sometimes I think it's good for me to leave home for a while.
45.	I'd rather stay silent than discuss with my spouse.
46.	Even if I'm right in the discussion, I stay silent to hurt my spouse.
47.	When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
48.	I feel right in our discussions.
49.	I have nothing to do with what I've been accused of.
50.	I'm not actually the one who's guilty about what I'm accused of.
51.	I'm not the one who's wrong about problems at home.
52.	I wouldn't hesitate to tell my spouse about her/his inadequacy.
53.	When I discuss, I remind my spouse of her/his inadequacy.
54.	I'm not afraid to tell my spouse about her/his incompetence.

But let's see how well we can predict with the engineered new features and decide what to do with these other questions (cause maybe we don't need them).


We now have a few additional features with consolidated information to help us simplify the analysis, which I'll copy to a new dataframe for analysis


```python
# Copying the data to a new dataframe
divorce_new = divorce.iloc[:,54:]
divorce_new.head()
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
      <th>Divorce_Y_N</th>
      <th>accept_difference</th>
      <th>alignment</th>
      <th>marriage_concept</th>
      <th>know_each_other</th>
      <th>agressive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2.25</td>
      <td>0.500000</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>1.285714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4.00</td>
      <td>3.166667</td>
      <td>3.4</td>
      <td>1.1</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2.00</td>
      <td>2.833333</td>
      <td>2.8</td>
      <td>1.7</td>
      <td>1.714286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2.50</td>
      <td>3.333333</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1.50</td>
      <td>0.500000</td>
      <td>1.2</td>
      <td>0.8</td>
      <td>0.571429</td>
    </tr>
  </tbody>
</table>
</div>



### Train and evaluate model



```python
# Separate features and labels
features = ['accept_difference','alignment','marriage_concept','know_each_other','agressive']
label = 'Divorce_Y_N'
X, y = divorce_new[features].values, divorce_new[label].values
```


```python
from matplotlib import pyplot as plt
%matplotlib inline

for col in features:
    divorce_new.boxplot(column=col, by=label, figsize=(6,6))
    plt.title(col)
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_21_0.png)
    



    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_21_1.png)
    



    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_21_2.png)
    



    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_21_3.png)
    



    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_21_4.png)
    



```python
# Split into train test groups
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
```

    Training cases: 119
    Test cases: 51
    


```python
# Train the model using a simple logistic regression
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(X_train, y_train)
print (model)
```

    LogisticRegression(C=100.0, max_iter=10000)
    

#### Let's predict


```python
predictions = model.predict(X_test)
```

#### And evaluate


```python
from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(y_test, predictions))

from sklearn. metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.metrics import precision_score, recall_score

print("Overall Precision:", precision_score(y_test, predictions, average='weighted'))
print("Overall Recall:", recall_score(y_test, predictions, average='weighted'))

```

    Accuracy:  0.9803921568627451
                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98        27
               1       0.96      1.00      0.98        24
    
        accuracy                           0.98        51
       macro avg       0.98      0.98      0.98        51
    weighted avg       0.98      0.98      0.98        51
    
    Overall Precision: 0.9811764705882353
    Overall Recall: 0.9803921568627451
    


```python
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# calculate ROC curve
y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/divorce_prediction_files/divorce_prediction_28_0.png)
    



```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
```

    AUC: 0.9984567901234569
    

In conclusion, it seems that a combination of information in the following area:
1. couples' ability to handle differences in opinions
2. the alignment of their life goals
3. similarity in their understanding of the meaning of marriage
4. how well they know each other
5. how agressive they could be in their communication 

could serve as a pretty accurate indication of the result of their marriage. I guess if you have questions and doubts in your marriage or are considering getting into marraige, this questionnaire might be able to help shed some light on where your relationship could be going...probably more accurate than [horoscope](https://genus.springeropen.com/articles/10.1186/s41118-020-00103-5).
