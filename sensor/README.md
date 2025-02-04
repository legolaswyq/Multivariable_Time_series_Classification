##  Dataset Description

In this competition, you'll classify 60-second sequences of sensor data, indicating whether a subject was in either of two activity states for the duration of the sequence.

## Files and Field Descriptions

- train.csv

   

  \- the training set, comprising ~26,000 60-second recordings of thirteen biological sensors for almost one thousand experimental participants

  - `sequence` - a unique id for each sequence
  - `subject` - a unique id for the subject in the experiment
  - `step` - time step of the recording, in one second intervals
  - `sensor_00` - `sensor_12` - the value for each of the thirteen sensors at that time step

- train_labels.csv

   

  \- the class label for each sequence.

  - `sequence` - the unique id for each sequence.
  - `state` - the state associated to each sequence. This is the target which you are trying to predict.

- **test.csv** - the test set. For each of the ~12,000 sequences, you should predict a value for that sequence's `state`.

- **sample_submission.csv** - a sample submission file in the correct format.



Kaggle Competition submit to see the Test score 

```bash
https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/data
```





# Env Setup 

```bash
conda env create -f environment.yml
```





1. EDA analyze the dataset 
2. feature selection, recursively 
3. PCA
4. hot map, feature correlation 

merge with label and see which feature has correlation to the label

```python
train = pd.merge(train, train_labels,how='left', on="sequence")

plt.figure(figsize=(25,8))
sns.heatmap(data.corr(), annot=True, cbar=True, cmap="YlGnBu")
```



1. feature engineer, min, max. mean, shift(1), diff 

```python
def addFeatures(df):  
    for feature in features:
        df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']    
    return df

train = addFeatures(train)
test = addFeatures(test)
```



1. remove duplicate, fillna (linear)

