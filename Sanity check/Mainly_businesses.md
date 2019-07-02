

```python
import pandas as pd
```


```python
# Pre-processing. All the same steps as was done on test data set

unclassified = pd.read_csv("businesses.csv", encoding ="latin1")

unclassified = unclassified[unclassified.face_detection != 999]

array = []
for i in list(unclassified["url"]):
    if str(i) != "nan":
        array.append(1)
    else:
        array.append(0)
unclassified["url_detect"] = array

unclassified = unclassified[(unclassified.color_number != 184) & (unclassified.color_number !=0)]
unclassified = unclassified.loc[(unclassified["label"] == "individual") | (unclassified["label"] == "business"), :]

unclassified = unclassified[["label", "username", "face_detection", "color_number", \
                             "b_key_count", "i_key_count", "url_detect"]]

unclassified_X = unclassified.drop(["label", "username"], axis=1)
y = unclassified["label"]
```


```python
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Step 1: Label-encode data set (y_train was previously "business" or "individual", which needed to be numbers) 
label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_y = label_encoder.transform(y)


# Step 2: Convert encoded labels to one-hot-encoding
y_categorical = to_categorical(encoded_y)
y_categorical.shape
```

    Using TensorFlow backend.





    (595, 2)




```python
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit(unclassified_X)
```

    /Users/kyleeng/anaconda3/envs/PythonData/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)



```python
X_scaled = X_scaler.transform(unclassified_X)
```

    /Users/kyleeng/anaconda3/envs/PythonData/lib/python3.7/site-packages/ipykernel_launcher.py:1: DataConversionWarning: Data with input dtype int64 were all converted to float64 by StandardScaler.
      """Entry point for launching an IPython kernel.



```python
# Load the model
from tensorflow.keras.models import load_model
twitter_normal_neural_trained = load_model("../twitter_normal_neural_trained.h5")
```

    WARNING:tensorflow:From /Users/kyleeng/anaconda3/envs/PythonData/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /Users/kyleeng/anaconda3/envs/PythonData/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.



```python
# unclassified_predictions = model.predict_classes(X_scaled)
# prediction_labels = label_encoder.inverse_transform(unclassified_predictions)
```


```python
model_loss, model_accuracy = twitter_normal_neural_trained.evaluate(
    X_scaled, y_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")
```

     - 0s - loss: 1.8486 - acc: 0.1580
    Normal Neural Network - Loss: 1.8485625040631335, Accuracy: 0.15798319876194



```python
# Something wrong... not sure what. Model needs to be trained on more data?
```
