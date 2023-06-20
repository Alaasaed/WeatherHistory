
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from random import seed
from random import random
from random import uniform
from math import exp
import seaborn as sns 
import scipy.stats as stats
# (i) Perform all necessary preprocessing for the given dataset
weather_df= pd.read_csv('weather_df.csv')
X = weather_df.copy() #dataset has been copied to  X
X.shape 
X.columns
X.head()
X.tail()  
X.describe() 
X.describe(include='all')
X.info()
###############################################################################################################################

                                  #(i) Perform all necessary preprocessing for the given dataset
#1) MISSING VALUE 
#identify the Missing Values in the data each columns
 
weather_df.isnull().any() # For the verifiy
X.isnull().sum() 
#Checking the number of nulls in percentage (517/96453)*100 
round(100*(weather_df.isnull().sum()/len(weather_df.index)),2)
weather_df['Precip Type'].value_counts() # To get the count of each available types
#Now we will try to impute null with the maximum occured values
#rain before  =85224 #rain after  =85541 #num of rain > num of snow so null cells will implent bu rain 
X.loc[X['Precip Type'].isnull(),'Precip Type']='rain'
#X = X.dropna() #drop all nan values in given data set
#################################################################################
#2) DUPLICATED ROW
#check for the duplicate row in data set.
print(X.duplicated().value_counts()) 
#this data set have 24 duplicated values.
print(X[X.duplicated()]) # To check view the duplicated values
X=X.drop_duplicates() # To drop the duplicate values 

################################################################################
#3) THE OUTLIERS

X._get_numeric_data().columns.tolist()
# make the boxplot of given columns to check the outliers
          ##################################################################
                     #Box plot and Q-Q Plot of Temperature © column
temp_df = pd.DataFrame(X, columns=['Temperature (C)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Temperature (C)"], dist="norm", plot=plt)
plt.show()
          ##################################################################
                #  Box plot and Q-Q Plot of Apparent Temperature  column
temp_df = pd.DataFrame(X, columns=['Apparent Temperature (c)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Apparent Temperature (C)"], dist="norm", plot=plt)
plt.show()
          ##################################################################
                # Box plot and Q-Q Plot of Wind Speed (km/h)  column
temp_df = pd.DataFrame(X, columns=['Wind Speed (km/h)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Wind Speed (km/h)"], dist="norm", plot=plt)
plt.show()
          ##################################################################

              # Box plot and Q-Q Plot of Wind Bearing (degrees)  column
temp_df = pd.DataFrame(X, columns=['Wind Bearing (degrees)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Wind Bearing (degrees)"], dist="norm", plot=plt)
plt.show()
         ##################################################################
              # Box plot and Q-Q Plot of Visibility (km)   column
temp_df = pd.DataFrame(X, columns=['Visibility (km)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Visibility (km)"], dist="norm", plot=plt)
plt.show()
            
        ##################################################################
        
                # Box plot and Q-Q Plot of Loud Cover   column
temp_df = pd.DataFrame(X, columns=['Loud Cover'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Loud Cover"], dist="norm", plot=plt)
plt.show()
# all values are same and zero. So we can drop that column from our data set

X['Loud Cover'].value_counts()#all values zero 
X = X.drop('Loud Cover', axis = 1)
            ##################################################################

                # Box plot and Q-Q Plot of Pressure (millibars)  column
temp_df = pd.DataFrame(X, columns=['Pressure (millibars)'])
temp_df.boxplot(vert=False)
#plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X["Pressure (millibars)"], dist="norm", plot=plt)
plt.show()
#there is  some anomalies.
X['Pressure (millibars)'].value_counts() 
#get the descriptions of given column such as mean, count, std, min etc.
X['Pressure (millibars)'].describe()
#to check the pattern of value changes,
plt.rcParams["figure.figsize"] = (200, 60)
plt.plot(X['Pressure (millibars)'].tolist(), label="Pressure")
plt.show()
#In above graph there are considerable amounts of pressure drops = zero
#there is no possibility to have zero values or less that zero values in that column
#calculate the percentage of the zero or less than zero values in that given column.
round (len(X[(X['Pressure (millibars)']<=0.0  )])   * 100/len(X),2)
# Pressure have very more outliers.So we can't drop those value rows 
#we can reset the index of data frames to avoid problems in future steps.
X=X.reset_index(drop=True)
# assign np.nan for zero values in pressure column to assign suitable values,
X.loc[X.index[X['Pressure (millibars)']<=0.0].tolist(), ['Pressure (millibars)']] =np.nan
#check whether we have assigned values correctly
X['Pressure (millibars)'].isnull().sum() # Verification
# we can assign suitable value to np.nan values in pressure column,
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer.fit(X[['Pressure (millibars)']])
X['Pressure (millibars)']=imputer.fit_transform(X[['Pressure (millibars)']])

#X.loc[X['Pressure'].isnull(),'Pressure'=''
#X.loc[X.index[X['Pressure (millibars)']=(X['Pressure (millibars)'].isnull()]).tolist(), ['Pressure (millibars)']] = '1016.814483'

           ##################################################################
                #Box plot and Q-Q Plot of Humidity column

# the boxplot of given columns to check the outliers,
temp_df = pd.DataFrame(X, columns=['Humidity'])
temp_df.boxplot(vert=False)
# plot the probability plot to given columns for check the patterns of distribution, values range and etc.,
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(X['Humidity'], dist="norm", plot=plt)
plt.show()
#From plot,some values are having 0.But practically is not possible to have zero values or less than that values on normal environment.
#the graph to see the values changes pattern
plt.rcParams["figure.figsize"] = (200, 60)
plt.plot(X['Humidity'].tolist(), label="Humidity")
plt.show()

#calculate the percentage of the zero or less than zero value
round (len(X[(X['Humidity']<=0.0) ])* 100/len(X),2)
#humidity have very less outliers.So we can drop those value rows in our data set.
X.drop(X[X['Humidity'] == 0].index, inplace = True)
#reset the index after each drop of row values to avoid some errors in future
X=X.reset_index(drop=True)
###############################################################################
#Data  handling
split = np.random.rand(len(X)) < 0.7
#return every index for which the split value is True
trainingSet = X[split]
# ~split means "not equal to" in df indexing.
testingSet = X[~split]
###########################################################################
#discretization 
#for ‘Wind Bearing (degrees)’ Because ‘Wind Bearing (degrees)’ have values from 0 to 359 degrees.
#Wind directions n_bins = 16 
"""
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform') 
discretizer.fit(X[['Wind Bearing (degrees)']])
X['Wind Bearing (degrees)'] = discretizer.transform(X[['Wind Bearing (degrees)']])
X['Wind Bearing']=labelencoder.fit_transform(X['Wind Bearing (degrees)'])
X = X.drop('Wind Bearing (degrees)', axis = 1)
X.head() # For the verification
"""
##############################################################################
#Data Standadization We have to remove the Categorical Features 
#after using preproceesing and PCA 
X.drop(["Formatted Date", "Summary","Precip Type"], axis = 1, inplace=True)
X.columns
########################################################################################################################################3
              #(ii)Suggest the MLNN architecture 
                                           
def initialize_network(n_inputs, n_hidden, n_outputs):    # Initialize a network
  network = list()
  hidden_layer = [];
  neuron= {'weights':[]}
  for i in range(n_hidden):
       for j in range(n_inputs+1):
           neuron['weights'].append(random());
       hidden_layer.append(neuron)     # W of hidden layer
       network.append(hidden_layer)

     
  output_layer =[];
  neuron= {'weights':[]}
  for i in range(n_outputs):
       for j in range(n_hidden + 1):
           neuron['weights'].append(random());
       output_layer.append(neuron)
       network.append(output_layer)    # W of output
  return network

# activation function
def activate(weights, inputs):
    activation = weights[-1] 
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] 
        print('inputs',inputs)
        print('activation',activation)
        input('In activation Cont')
    return activation
#transfer function

def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
#forword propagation
def forward_propagate(network, Row):
    inputs = Row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)		
            new_inputs.append(neuron['output']) 
            inputs = new_inputs
            print("update inputs: ",inputs)
      
    return inputs     

# Back word propagate 
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
      
        #---error calculations
        if i != len(network)-1:
            for j in range(len(layer)): 
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else: 
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                
                
    #---Delta calculation
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer(neuron['output'])
########################################################################################################################################


    # (iii)Randomly set the weights’ values. .
weights = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1),
np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
bias = np.random.uniform(0, 1)
numberOfEpochs = 10
epochSize = int((len(trainingSet)) / numberOfEpochs)
learningRate = 0.0005
epochStart = 0
epochEnd = epochSize

epoch = trainingSet.iloc[epochStart:epochEnd]
desiredOutput = epoch.iloc[:, 2]
epoch = epoch.drop('Humidity', axis='columns',)
#Return a new array of shape:epochSize and type:float
new_array = np.empty(epochSize, float)
#Return a new array error of shape:epochSize and type:float.
error = np.empty(epochSize, float)
deltaWeights = np.empty((epochSize, 7), float)
deltaBias = np.empty(epochSize, float)


for k in range(numberOfEpochs):
    for i in range(epochSize):
        new_array[i] = np.dot(epoch.iloc[i], weights)+ bias
        error[i] = new_array[i] - desiredOutput.iloc[i]
        deltaWeights[i] = np.dot(epoch.iloc[i], error[i] * learningRate)
        deltaBias[i] = learningRate * error[i]
              #update the weights
    weights = weights - np.mean(deltaWeights)
             #update the biase
    bias = bias - np.mean(deltaBias)
    epochStart = epochStart + epochSize
    epochEnd = epochEnd + epochSize
    epoch = trainingSet.iloc[epochStart:epochEnd]
    desiredOutput = epoch.iloc[:, 2]
    epoch = epoch.drop('Humidity', axis='columns')


epochSize = int((len(testingSet)) / numberOfEpochs)
epochStart = 0
epochEnd = epochSize

testEpoch = testingSet.iloc[epochStart:epochEnd]
realOutput = testEpoch.iloc[:, 2]
testEpoch = testEpoch.drop('Humidity', axis='columns')

meanSquareError = np.empty(numberOfEpochs, float)
predictedOutput = np.empty(epochSize, float)
E = np.empty(epochSize, float)
squareError = np.empty(epochSize, float)


#compute the MSE error
for k in range(numberOfEpochs):
    for i in range(epochSize):
        predictedOutput[i] = np.dot(testEpoch.iloc[i], weights) + bias
        E[i] = predictedOutput[i] - realOutput.iloc[i]
        squareError[i] = (E[i]) ** 2

    meanSquareError[k] = np.mean(squareError)

    epochStart = epochStart + epochSize
    epochEnd = epochEnd + epochSize
    testEpoch = testingSet.iloc[epochStart:epochEnd]
    realOutput = testEpoch.iloc[:, 2]
    testEpoch = testEpoch.drop('Humidity', axis='columns')

print(meanSquareError)

# draw a chart with a horizontal axis titled “Epoch number” ranging from 1 to 10, and a vertical axis titled “MSE”.
plt.plot(meanSquareError)
plt.suptitle('Error Calculation')
plt.show()
















































