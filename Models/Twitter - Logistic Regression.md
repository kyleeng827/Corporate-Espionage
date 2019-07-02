
includes everything except words analysis, 6 hidden nodes, deep learning less acurate, loss= categorical_crossentropy

without following and followers


```python
import pandas as pd
```


```python
df = pd.read_csv("final_output.csv", encoding = "latin1")
```


```python
df.head()
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
      <th>label</th>
      <th>id</th>
      <th>name</th>
      <th>username</th>
      <th>bio</th>
      <th>location</th>
      <th>url</th>
      <th>join_date</th>
      <th>join_time</th>
      <th>tweets</th>
      <th>...</th>
      <th>media</th>
      <th>private</th>
      <th>verified</th>
      <th>profile_image_url</th>
      <th>background_image</th>
      <th>color_number</th>
      <th>face_detection</th>
      <th>follower/following_ratio</th>
      <th>b_key_count</th>
      <th>i_key_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>individual</td>
      <td>379823034</td>
      <td>mohammed</td>
      <td>BarcaMohammed</td>
      <td>NaN</td>
      <td>North Bergen</td>
      <td>NaN</td>
      <td>9/25/11</td>
      <td>9:33:00</td>
      <td>172.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>https://pbs.twimg.com/profile_images/166601617...</td>
      <td>NaN</td>
      <td>51912.0</td>
      <td>0.0</td>
      <td>0.023256</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>individual</td>
      <td>931229576</td>
      <td>SaMisches</td>
      <td>Shmisch</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11/6/12</td>
      <td>20:41:00</td>
      <td>6.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>https://pbs.twimg.com/profile_images/930857677...</td>
      <td>NaN</td>
      <td>31033.0</td>
      <td>0.0</td>
      <td>0.108696</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>individual</td>
      <td>393095898</td>
      <td>Christopher Pawelski</td>
      <td>ChrisPawelski</td>
      <td>A 4th generation family onion farmer married t...</td>
      <td>Florida, NY</td>
      <td>http://about.me/chris_pawelski</td>
      <td>10/17/11</td>
      <td>18:00:00</td>
      <td>49503.0</td>
      <td>...</td>
      <td>23400.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>https://pbs.twimg.com/profile_images/686658233...</td>
      <td>https://pbs.twimg.com/profile_banners/39309589...</td>
      <td>46205.0</td>
      <td>0.0</td>
      <td>1.085431</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>individual</td>
      <td>3109060051</td>
      <td>xrishin</td>
      <td>xrishin</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3/26/15</td>
      <td>19:39:00</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>https://abs.twimg.com/sticky/default_profile_i...</td>
      <td>NaN</td>
      <td>184.0</td>
      <td>0.0</td>
      <td>0.050000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>business</td>
      <td>1.13E+18</td>
      <td>LENDVER</td>
      <td>LendverLLC</td>
      <td>BORROW CONFIDENTLY. We__Î¢_öÎå_Î¢ve do...</td>
      <td>NaN</td>
      <td>https://www.lendver.com/</td>
      <td>5/29/19</td>
      <td>7:32:00</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>https://pbs.twimg.com/profile_images/113375859...</td>
      <td>https://pbs.twimg.com/profile_banners/11337428...</td>
      <td>510.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df.shape
```




    (11456, 23)




```python
# 999 was a code we used when executing python face_detection library on twitter profile pictures. 
# Specifically, 999 indicated that there was an error/no picture found when iterating through thr profile pictures
df = df[df.face_detection != 999]
# code 1 was assigned if a face was deteced, code 0 was assigned if a face was not detected.
```


```python
# Create a column in dataframe indicating whether a url was linked on the twitter page or not. 
# If a url was detected, code 1 was assigned, else, 0 was assigned.
array = []
for i in list(df["url"]):
    if str(i) != "nan":
        array.append(1)
    else:
        array.append(0)
df["url_detect"] = array
```


```python
df.shape
```




    (11264, 24)




```python
# color_number column is a column that gives a count of the number of colors a profile picture has.
# We noticed that pictures with 184 colors all linked to the same default twitter profile picture.
# Since no face recognition or insightful color count can be returned and properly evaluated by the model,
# we decided to drop all rows that have a color count of 184.
# Coincidentally, we noticed color_number = 0 all corresponding face_detection = 999.
# Even though that was already dropped, we decided to add this precaution in case this was used on future scraped data
df = df[(df.color_number != 184) & (df.color_number !=0)]
```


```python
df.shape
```




    (10467, 24)




```python
# Drop columns we dont need for modeling
# df = df.drop(columns = ["media", "name", "following", "followers", "id", "url", "bio", "location", "join_date", "join_time", \
#                         "private", "verified", "profile_image_url", "background_image", "i_key_count"])
df = df[["label", "username", "face_detection", "color_number", "b_key_count", "i_key_count", "url_detect"]]
```


```python
df = df.dropna(how="any")
# Dataset used had 11401 rows, unclassified as business or not. Our team went into each twitter page to classify
# whether the page was a business or individual. Since 1000s of rows could not be manually classified, 
# we dropped many unclassified rows
# Dropping data must be the last step, because there are blank fields in the dropped columns, which may have values
# in the variables we will take into consideration. Dropping pre-emptively will drop many rows unnecessarily
```


```python
df.shape
```




    (2357, 7)




```python
df.head()
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
      <th>label</th>
      <th>username</th>
      <th>face_detection</th>
      <th>color_number</th>
      <th>b_key_count</th>
      <th>i_key_count</th>
      <th>url_detect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>individual</td>
      <td>BarcaMohammed</td>
      <td>0.0</td>
      <td>51912.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>individual</td>
      <td>Shmisch</td>
      <td>0.0</td>
      <td>31033.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>individual</td>
      <td>ChrisPawelski</td>
      <td>0.0</td>
      <td>46205.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>business</td>
      <td>LendverLLC</td>
      <td>0.0</td>
      <td>510.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>individual</td>
      <td>ClaretSport</td>
      <td>1.0</td>
      <td>41897.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Assign X (data) and y (target)
X = df.drop(["label", "username"], axis=1)
y = df["label"]
print(X.shape, y.shape)
```

    (2357, 5) (2357,)



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
```


```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier
classifier.fit(X_train, y_train)

# Prediction results on test data set
predictions = classifier.predict(X_test)
comparison = pd.DataFrame({"prediction": predictions, "actual": y_test})
comparison.head()
```

    /Users/kyleeng/anaconda3/envs/PythonData/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





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
      <th>prediction</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10300</th>
      <td>individual</td>
      <td>business</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>10284</th>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>2494</th>
      <td>individual</td>
      <td>business</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>individual</td>
      <td>individual</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
```

    Training Data Score: 0.6468590831918506
    Testing Data Score: 0.6474576271186441



```python
# joined dataset so we can see usernames and predictions/actual side by side
merged_comparsion = df.join(comparison, how ="inner")
merged_comparsion = merged_comparsion[["username", "prediction", "actual"]]

merged_comparsion.head()
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
      <th>username</th>
      <th>prediction</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>ChrisPawelski</td>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arvidinvest</td>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>17</th>
      <td>pattyshort</td>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HarshnaParoha</td>
      <td>individual</td>
      <td>individual</td>
    </tr>
    <tr>
      <th>23</th>
      <td>jsare34</td>
      <td>individual</td>
      <td>individual</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train, y_train)
rf.score(X_train, y_train)
```




    0.9954725523486134




```python
# Get weights that machine defined, multiplied by 100 to get percents
importances = rf.feature_importances_*100
importances
```




    array([29.74523888, 54.15080507,  7.30441884,  1.68413465,  7.11540257])




```python
sorted(zip(importances, X), reverse=True)
```




    [(54.15080506755916, 'color_number'),
     (29.745238875813087, 'face_detection'),
     (7.304418836684552, 'b_key_count'),
     (7.115402565934826, 'url_detect'),
     (1.6841346540083724, 'i_key_count')]




```python

```
