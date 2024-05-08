# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
    data=pd.read_csv("C:/Users/USER/Documents/GAP Prediction.csv")

# %%
data.head()

# %%
data.tail()

# %%
data.isnull().sum()

# %%
data.hist(bins=5)

# %%
data.corr()

# %%
#allocate SAT to x and GPA to Y
X=data["SAT"].values
Y=data["GPA"].values

# %%
X

# %%
Y

# %%
#to check the rows and column
X.shape
Y.shape

# %%
#scatter plot graph 
#we need to pass X and Y value
sns.scatterplot(x=X,y=Y, color='blue')
plt.xlabel("SAP")
plt.ylabel("GPA")

# %%
#we need to train a data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

# %%
xtrain

# %%
X

# %%
xtrain.shape[0]

# %%
X.shape[0]

# %%
print("xtrain",xtrain.shape)
print("xtest",xtest.shape)
print("ytrain",ytrain.shape)
print("ytest",ytest.shape)

# %%
#we do this bcoz the linear regression works only in 1D array
ytrain = ytrain.reshape(-1,1)  # or ytrain.ravel()
xtrain = xtrain.reshape(-1,1)
ytest = ytest.reshape(-1,1)
xtest = xtest.reshape(-1,1) 

# %%
xtrain

# %%
from sklearn.linear_model import LinearRegression

# %%
lr=LinearRegression()

# %%
model=lr.fit(xtrain,ytrain)

# %%
ypredict=model.predict(xtest)

# %%
ytest

# %%
ypredict

# %%
#plot LinearRegression graph
sns.scatterplot(x=X,y=Y,color='r')
plt.plot(xtest,ypredict)
plt.xlabel("SAT")
plt.ylabel("GPA")

# %%
#Evaluation (MAE,MSE,RMSE from module called metrics)
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
#MAE-  Use when you want to know the average size of errors. It's easy to understand and not greatly affected by outliers.
#MSE-  Use when you want larger errors to have more impact. It squares errors, so big mistakes are penalized more than small ones.
#RMSE - Similar to MSE but gives error in the same units as the target variable. Useful for understanding error size in a meaningful context.
#R2 - How well the model fits the data
print("Mean Absolute error - ", mean_absolute_error(ytest,ypredict))

# %%
print("Mean Squared Erro - ", mean_squared_error(ytest,ypredict))

# %%
print("root_mean_squared_error - ", root_mean_squared_error(ytest,ypredict))

# %%
from math import sqrt
print("root_mean_squared_error - ",sqrt(mean_squared_error(ytest,ypredict)))

# %%
#R2 is for to check how well the model fits the data
r2=r2_score(ytest,ypredict)
r2

# %%
#if we have more independent variable in our dataset then we have to use adjusted R2

# %%
model.predict([[1500]])

# %%
import pickle

# %%
X

# %%
Y

# %%
def gradientdescent(x,y):
    a1_curr=0
    a0_curr=0
    it=1000
    n=len(x)
    learning_rate=0.01
    for i in range(it):
        ypredict=a1_curr * x + a0_curr
        cost=(1/n)* sum([val**2 for val in (y-ypredict)])
        a1_d=(2/n)*sum(x*(y-ypredict))
        a0_d=(2/n)*sum(y-ypredict)
        a1_curr=a1_curr-learning_rate*a1_d
        ypredict_gradient_descent = a1_curr * X + a0_curr
        a0_curr=a0_curr-learning_rate*a0_d
        print("a1  {}, a0  {}, cost  {}, iteration {}".format(a1_curr,a0_curr,cost,i))

# %%
gradientdescent(X,Y)

# %%
# Call gradient descent function
gradientdescent(X, Y)

# Calculate predicted values using the updated parameters


# Evaluate R2 score
r2_gradient_descent = r2_score(Y, ypredict_gradient_descent)
print("R2 score after gradient descent:", r2_gradient_descent)


# %%



