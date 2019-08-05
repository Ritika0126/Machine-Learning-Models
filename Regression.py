#Calculating Correlation Coefficient
def find_correlation(x, y):
    # Find the length of the lists
    n = len(x)

    # Find the sum of the products
    products = []
    for xi, yi in zip(x, y):
        products.append(xi * yi)
    sum_products = sum(products)

    # Find the sum of each list 
    sum_x = sum(x) 
    sum_y = sum(y)

    # Find the squared sum of each list 
    squared_sum_x = sum_x ** 2 
    squared_sum_y = sum_y ** 2

    # Find the sum of the squared lists
    x_square = []
    for xi in x:
        x_square.append(xi ** 2)
    x_square_sum = sum(x_square)

    y_square = []
    for yi in y:
        y_square.append(yi ** 2)
    y_square_sum = sum(y_square)

    # Use formula to calculate correlation
    numerator = n * sum_products - sum_x * sum_y
    denominator1 = n * x_square_sum - squared_sum_x
    denominator2 = n * y_square_sum - squared_sum_y
    denominator = (denominator1 * denominator2) ** 0.5
    correlation = numerator / denominator
    return correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading data
data = pd.read_csv('AppleStore.csv')
print(data.head())

#Data pre-processing
price = list(data['price'].values)
k = list(data.columns)
print(k)
X=[]
for i in k:
    if(i!='price') and (i!='currency') and (i!='id') and (i!='track_name') and (i!='unnamed: 0'):
        if data[i].dtype==object:
            un = np.unique(data[i].values)
            print(len(list(un)))
            temp=[]
            for ind1,p in enumerate(list(data[i].values)):
                for ind2,q in enumerate(list(un)):
                    if q==p:
                        temp.append(ind2)
            X.append(temp)
        else:
            X.append(list(data[i]))
print(X[8][:10])

                    
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

#For Training and Testing data split
from sklearn.model_selection import train_test_split, cross_val_score

# X and Y Values
X = np.array(X).T
Y = np.array(price)
#print(Y)


#Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4)
#print(X_train[0],X_test[0])

# Model Intialization
reg = LinearRegression()
#print(type(reg))
#t = tuple(1000 for x in range(1000))
#NN = MLPRegressor(learning_rate_init = 0.001, hidden_layer_sizes = (100,100))
NN = MLPRegressor(max_iter = 10000, solver = 'adam', learning_rate = 'constant', learning_rate_init = 0.001, activation = 'tanh')
neigh = KNeighborsRegressor(n_neighbors=100)

# Data Fitting
reg = reg.fit(X_train, y_train)
#print(type(reg))
NN = NN.fit(X_train, y_train)
#print(type(NN))
#print(NN.n_layers_)
KNN = neigh.fit(X_train, y_train)



# Y Prediction
Y_pred = NN.predict(X_test)
Y_pred_KNN = KNN.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, Y_pred_KNN))
r2 = neigh.score(X_test, y_test)
r2s = r2_score(y_test, Y_pred_KNN)
r =  find_correlation(y_test, Y_pred_KNN)

#Cross Validation
scores_NN = cross_val_score(NN, X, Y, cv=5)
scores_LM = cross_val_score(reg, X, Y, cv=5)
scores_KNN = cross_val_score(neigh, X, Y, cv=5)
#print("Cross Validation score of KNN is:",scores_KNN.mean())


#Evaluation Metrics
print(rmse)
#print(r2)
print(r**2)
print(r2s)

#Plot between predictions and actual values
plt.plot(y_test, Y_pred_KNN, 'o')
plt.show()


#Calculating coefficients
#print(list(reg.coef_))
#print(NN.coefs_)

#Comparing models
import matplotlib.pyplot as plt
def create_bar_chart(data, labels):
    num_bars = len(data) # Number of bars to draw
    positions = range(1, num_bars+1) # Positions of bars on y-axis
    plt.bar(positions, data) # Generate the bar chart
    plt.xticks(positions, labels) # Add little markers to each label on x-axis
    plt.xlabel('Models') # At x-axis label
    plt.ylabel('Coefficient of Determination') # Add y-axis label
    plt.title('Comparison of machine learning models on Apple Store Data Problem') # Add title
    plt.grid()  #Add a grid for easier visual estimation of values
    plt.show()

create_bar_chart([scores_NN.mean(), scores_LM.mean(), scores_KNN.mean()], ['Neural Networks', 'Linear Regression', 'K-Nearest Neighbors'])

