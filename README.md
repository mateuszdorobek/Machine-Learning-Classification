# Machine Learning - Classification Algorithms
Mateusz Dorobek

## Project
Main goal of this project is to compare and find the best algorithm for binary classification problem.

## Data

Data used in this project comes from one of telecomunication company. They were anonimized, so data preparation couldn't base on expert experience. 
All data can be found at this [link](https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/index.html). 
All preprocessing, including standardization, encoding, factorization and other techniques was done using **Python**.
Preprocessing script as well as this file is on my *GitHub* account at this [link](https://github.com/SaxMan96/Machine-Learning-Classification/blob/master/data_extraction.ipynb)

I decided that if column has more than 99% of NaNs I'll remove it, so 23 columns went off.
I've concatenated train and test set to perform the same data modyfications on both of them.
```python
def load_data() -> pd.Series:
    csv_train = pd.read_csv('train.txt', sep=" ").assign(train = 1) 
    csv_test = pd.read_csv('testx.txt', sep=" ").assign(train = 0)
    csv = pd.concat([csv_train,csv_test])
    return csv
```
Very usefull was `stat` function which write a lot of usefull statistics about column on screen
```python
def stat(f):
    nans = nans_ctr()
    unique = unique_ctr()
    val_type = val_types()
    print(f"min: {csv[f].min()}")
    print(f"max: {csv[f].max()}")
    print(f"nans: {nans[f]}")
    print(f"unique: {unique[f]}")
    print(f"val_type: {val_type[f]}")
    print(f"vals per class: {round((len(csv)-nans[f])/unique[f],2)}")
```
### Data Processing

I've used `threshold_factorization` on data, where was a lot of text values not connected eachother, so I've decided to calculate quantity and split them in several groups based on their number.
```python
def threshold_factorization(data, *t_list) -> pd.Series():
   letter_counts = Counter(data)
      df = pd.DataFrame.from_dict(letter_counts, orient='index')
      df = df.sort_values(by=0, ascending=False)
      t_list = (df.values[0].item()+1,) + t_list + (0,)
      out = data.copy()
      for i in tqdm(range(1,len(t_list)),desc="Progress",leave=False):
          idx = df[(df>t_list[i]).values & (df<=t_list[i-1]).values].index
          for j in tqdm(idx,leave=False):
              out.loc[out == j] = i
      return out
```
Example of such distribution

# Algorithms

To perform classification and choose the best algorithm I've used R. Whole script you can find at my GitHub at this [link](https://github.com/SaxMan96/Machine-Learning-Classification/blob/master/classifiers.R)
I've used 20% of training data as validation set.

### Extremely Randomized Trees

<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/ET_Lift.png"  width="400"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/ET_ROC.png"  width="400"/>


### Random Forest

<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/RF_Lift.png"  width="400"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/RF_ROC.png"  width="400"/>

### XGBoost

<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/XGB_Lift.png"  width="400"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/XGB_ROC.png"  width="400"/>


### Generalized Boosted Regression Modeling

<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/GBM_Lift.png"  width="400"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/GBM_ROC.png"  width="400"/>

# Summarise

For final algoritm I've choosen Generalized Boosted Regression Modeling
<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/ALL_Lift.png"  width="600"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/ALL_ROC.png"  width="600"/>

|     Name      | Lift10 	| AUC 	 |
|:------------:	|:------:	|:---:	 |
|  ExtraTrees  	|  7.13  	| 78% 	 |
| RandomForest 	|  7.99  	| 81% 	 |
|    XGBoost   	|  8.87  	| 80% 	 |
|      **GBM**  |**9.93** | **81%**|
