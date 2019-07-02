

```python
import pandas as pd
```


```python
df = pd.read_csv("final_output.csv", encoding = "latin1")
```


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
# color_number column is a column that gives a count of the number of colors a profile picture has.
# We noticed that pictures with 184 colors all linked to the same default twitter profile picture.
# Since no face recognition or insightful color count can be returned and properly evaluated by the model,
# we decided to drop all rows that have a color count of 184.
# Coincidentally, we noticed color_number = 0 all corresponding face_detection = 999.
# Even though that was already dropped, we decided to add this precaution in case this was used on future scraped data
df = df[(df.color_number != 184) & (df.color_number !=0)]
```


```python
# Drop columns we dont need for modeling
# df = df.drop(columns = ["name", "id", "url", "bio", "location", "join_date", "join_time", \
#                         "private", "verified", "profile_image_url", "background_image"])
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
# Assign X (data) and y (target)
X = df.drop(["label", "username"], axis=1)
y = df["label"]
print(X.shape, y.shape)
```

    (2357, 5) (2357,)



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
```


```python
# Note that k: 9 provides the best accuracy where the classifier starts to stablize
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print('k=2 Test Acc: %.3f' % knn.score(X_test, y_test))
```

    k=2 Test Acc: 0.769

