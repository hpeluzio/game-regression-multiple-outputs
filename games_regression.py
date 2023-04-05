import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
name_games = base.Name
base = base.drop('Name', axis = 1)

predictors = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
sale_na = base.iloc[:, 4].values
sale_eu = base.iloc[:, 5].values
sale_jp = base.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# changing some string values to numeric values
predictors[:, 0] = labelencoder.fit_transform(predictors[:, 0])
predictors[:, 2] = labelencoder.fit_transform(predictors[:, 2])
predictors[:, 3] = labelencoder.fit_transform(predictors[:, 3])
predictors[:, 8] = labelencoder.fit_transform(predictors[:, 8])

# Encoding numeric values to kind of binaries
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
predictors = onehotencorder.fit_transform(predictors).toarray()


# Structuring the neural network
input_layer = Input(shape = (61,))
# 61 entradas, 3 saidas... divide por 2
# (61 + 3) / 2 = 32
hidden_layer1 = Dense(units = 32, activation = 'sigmoid')(input_layer)
hidden_layer2 = Dense(units = 32, activation = 'sigmoid')(hidden_layer1)
output_layer1 = Dense(units = 1, activation = 'linear')(hidden_layer2)
output_layer2 = Dense(units = 1, activation = 'linear')(hidden_layer2)
output_layer3 = Dense(units = 1, activation = 'linear')(hidden_layer2)

regressor = Model(inputs = input_layer, 
                  outputs = [output_layer1, output_layer2, output_layer3])
regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(predictors, [sale_na, sale_eu, sale_jp], 
              epochs = 5000, batch_size = 100)

prediction_na, prediction_eu, prediction_jp = regressor.predict(predictors) 



