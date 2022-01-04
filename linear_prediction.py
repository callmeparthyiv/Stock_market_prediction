import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

#readind dataset
df = pd.read_csv('RELIANCE.csv')
df = df[['Open','High','Low','Close']]
print(df.head())

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

df =df.dropna() #dropping empty cells
X = df[['Open-Close','High-Low']]
print(X.head())
Y = np.where(df['Close'].shift(-1) > df['Close'],1,-1) 

split_percent = 0.7
split = int(split_percent*len(df))

x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
Y_test = Y[split:]

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

# #accuracy
# accuracy_train = accuracy_score(y_train,model.predict(x_train))
# accuracy_test = accuracy_score(Y_test,model.predict(x_test))


#predict signal
df['Predict_signal'] = model.predict(X)
df['Return'] = np.log(df['Close']/df['Close'].shift(1))

Cummulative_return = df[split:]['Return'].cumsum()*100

df['Strategy_return'] = df['Return']*df['Predict_signal'].shift(1)
Cummulative_strategy_return = df[split:]['Strategy_return'].cumsum()*100


#plotting
plt.figure(figsize=(10,5))
plt.plot(Cummulative_return,color='r',label = 'Return')
plt.plot(Cummulative_strategy_return,color = 'g',label = 'Strategy Return')
plt.legend()
plt.show()