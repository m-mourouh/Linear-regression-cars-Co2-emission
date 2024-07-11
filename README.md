# Importing necessary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

# Loading the dataset
```python
df = pd.read_csv("./cars_CO2_emission.csv")
```

# Selecting features
```python
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
data = df[features]
```

# Splitting data into training and test sets
```python
train, test = train_test_split(data, test_size=0.2, random_state=42)
x_train = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_train = train['CO2EMISSIONS']
x_test = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_test = test['CO2EMISSIONS']
```

# Training the linear regression model
```python
model = LinearRegression()
model.fit(x_train, y_train)
```

# Making predictions
```python
predictions = model.predict(x_test)
```

# Model coefficients and evaluation metrics
```python
print('Coefficients:', model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print('Variance score: %.2f' % r2_score(y_test, predictions))
```

# Plotting the distribution of CO2 Emissions
```python
plt.figure(figsize=(10, 6))
plt.hist(data['CO2EMISSIONS'], bins=30, color = "lightblue", ec="black")
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")
plt.title("Distribution of CO2 Emissions")
plt.show()
```

![1](https://github.com/m-mourouh/Linear-regression-cars-Co2-emission/assets/60442896/1fd2646d-5cf3-4970-adcb-605360031e17)


# Plotting the box plot of CO2 Emissions
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['CO2EMISSIONS'], color='lightgreen')
plt.xlabel("CO2 Emissions")
plt.title("Box Plot of CO2 Emissions")
plt.show()
```

![download](https://github.com/m-mourouh/Linear-regression-cars-Co2-emission/assets/60442896/68b6bda6-824c-4b0f-8296-c588bbf424e9)

# Plotting pair plot of features and CO2 Emissions
```python
sns.pairplot(data, diag_kind='kde', markers='+')
plt.suptitle("Pair Plot of Features and CO2 Emissions", y=1.02)
plt.show()
```
![Pair Plot of Features and CO2 Emissions](https://github.com/m-mourouh/Linear-regression-cars-Co2-emission/assets/60442896/072e411d-6de4-4a68-82e4-7e488614aae7)

# Plotting residuals
```python
plt.figure(figsize=(10, 6))
sns.residplot(x=predictions, y=y_test - predictions, lowess=True, color='hotpink')
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```
![download (1)](https://github.com/m-mourouh/Linear-regression-cars-Co2-emission/assets/60442896/ce75ed46-c347-4cd2-9f16-b9f8a03111a7)

