import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('IceCreamData.csv')


x_data = data[["Temperature"]]
y_data = data[["Revenue"]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

my_model = LinearRegression()

my_model.fit(x_train, y_train)

y_predict = my_model.predict(x_test)
print (y_predict)

plt.scatter(x_test, y_test, label='Actual Data' , color ='green')
plt.plot(x_test, y_predict, color='red', label='Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.legend()
plt.show()
