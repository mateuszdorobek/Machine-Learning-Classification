# Machine Learning - Classification Algorithms
## Mateusz Dorobek

## Project
Main goal of this project is to compare and find the best algorithm for binary classification problem.

## Data

Data used in this project comes from one of telecomunication company. They were anonimized, so data preparation couldn't base on expert experience. 
All data can be found at this [link](https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/index.html). 
All preprocessing, including standardization, encoding, factorization and other techniques was done using **Python**.
Preprocessing script as well as this file is on my *GitHub* account at this [link](https://github.com/SaxMan96/Machine-Learning-Classification/blob/master/data_extraction.ipynb)



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
<img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/All_Lift.png"  width="600"/><img src="https://raw.githubusercontent.com/SaxMan96/Machine-Learning-Classification/master/images/All_ROC.png"  width="600"/>

|     Name      | Lift10 	| AUC 	 |
|:------------:	|:------:	|:---:	 |
|  ExtraTrees  	|  7.13  	| 78% 	 |
| RandomForest 	|  7.99  	| 81% 	 |
|    XGBoost   	|  8.87  	| 80% 	 |
|      **GBM**  |**9.93** | **81%**|
