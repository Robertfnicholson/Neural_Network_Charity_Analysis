# Neural Network Charity Challenge

## Overview of the Analysis
In this challenge, I built and trained a deep learning neural network model in order to predict which organizations are worth donating to and which are too high risk. I used Python TensorFlow library to build and test the model. This robust deep learning neural network model is capable of interpreting large complex datasets. I used a dataset of more than 34,000 organizations that have received funding from my company, Alphabet Soup, and ran in the model after preprocessing the data. It now helps, our company, Alphabet Soup, determine which organizations should receive donations. My code can be found in “AlphabetSoupCharity.ipynb” and AlphabetSoupCharity_Optimization.ipynb.”

## Results
### Data Preprocessing
*	The target for my model is the column IS_SUCCESSFUL.
*	After eliminating EIN and NAME columns, the remaining columns are considered to be features for the model.
*	EIN, and “NAME” are neither targets nor features and should be removed from the input data. Also, those columns which are categorical variables had to be preprocessed for the model and the original columns dropped. These were APPLICATION TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and SPECIAL_CONSIDERATIONS.
*	I used the value_counts function, plot.density graph, and Python code to determine and to create binning for the APPLICATION and CLASSIFICATION columns.
*	Also used OneHotEncoder function and the fit_transform function to fit, transform and add the encoded columns to the dataframe.
*	I then merged the one-hot encoded features into the dataframe and dropped the original columns. A screenshot of the code is below:
![Preprocessing.png](https://github.com/Robertfnicholson/Neural_Network_Charity_Analysis/blob/0307b0ee3027d6d0817ee816841b1b969ef36015/Preprocessing.png)

![Preprocess_2.png](https://github.com/Robertfnicholson/Neural_Network_Charity_Analysis/blob/0307b0ee3027d6d0817ee816841b1b969ef36015/Preprocess_2.png)
### Compiling, Training, and Evaluating the Model
*	In the initial model, I selected three times the number of inputs for neurons, i.e., there were 45 inputs, so I used 135 neurons. I also selected an input layer, two input layers and an output layer since these are the minimum number of layers required for a deep learning, neural network model. 
*	The initial model resulted in an accuracy score of 65.9%, below the target of 75%.
*	To increase the model performance, I used automated model optimization that varied the activation function, the neurons in the first layer, and the number of hidden layers. This resulted in a 72.9% accuracy, which also was below the target of 75%.    
![Compile_Train_Evaluate.png](https://github.com/Robertfnicholson/Neural_Network_Charity_Analysis/blob/0307b0ee3027d6d0817ee816841b1b969ef36015/Compile_Train_Evaluate.png)

![Deep_Learning_Model.png](https://github.com/Robertfnicholson/Neural_Network_Charity_Analysis/blob/0307b0ee3027d6d0817ee816841b1b969ef36015/Deep_Learning_Model.png)

![ASC_OPT_v1.png](https://github.com/Robertfnicholson/Neural_Network_Charity_Analysis/blob/f3b158486f284e0ff5a6cfcfbc9bb65dbf4712e0/ASC_OPT_v1.png)

* I also ran two additional versions of the model: (1) one in which I eliminated the outliers for the ASK_AMT and (2) the other in which I eliminated the ASK_AMT column altogether. However, this did not result in increased accuracy. In fact the accuracy decreased when I eliminated the outliers for the ASK_AMT to 52.7%. 

## Summary
 I ran multiple versions of the deep learning model. These provided accuracy results that varied from 64.9% in the initial model to the optimized model of 72.9%. I recommend using a RandomForest model and compare these results to the neural network, deep learning model. In a RandomForest model both output and feature selection are easy to interpret, and it can easily handle outliers and nonlinear data. </p>

