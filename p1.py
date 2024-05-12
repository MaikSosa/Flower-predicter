"""
####################################################################################
#    Author: Mike Sosa                                                             #
#    Time spent: 8.1 hours                                                         #
#    Description: This program will read a dataset of flowers to predict           #
                  and analyze by the data provided what species it is by the size  #
                  of the data.                                                     #
####################################################################################
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


flowers = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/iris.csv')


sns.set_palette("Set1")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=flowers, x='petal_length', y='petal_width', hue='species')
plt.title('Scatter Plot of Petal Length vs. Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(title='Species')
plt.show()

# Create a new dataframe called X that contains the features we're going
# to use to make predictions
X = flowers[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X.head()

# Create a new dataframe called y that contains the target we're
# trying to predict
y = flowers['species']
y.head()

# The training data should contain 80% of the samples and
# the test data should contain 20% of the samples.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Create an instance of the model, configuring it to use the 3 nearest neighbors
# store the instance in a variable (always want to use the three closest
clss = KNeighborsClassifier(n_neighbors=3)

# Call the "fit" method of the classifier instance we created in step 4.
# Pass it the X_train and y_train data so that it can learn to make predictions
clss.fit(X_train, y_train)

# Use the predict() method to get a list of predictions for the samples in our
# test data. Then output those predictions
test_pred = clss.predict(X_test)


# Just a quick comparison with y_test to see if they match up
print(y_test)

# Import the accuracy_score function and use it to determine
# how accurate the models predictions were for our test data
accuracy_score(y_test, test_pred)
print(accuracy_score)

# Import the confusion_matrix function and use it to generate a confusion
# matrix of our model results.
matrix = confusion_matrix(y_test, test_pred)
print(matrix)


# Create a Seaborn heatmap
labels = np.unique(y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=20)
plt.show()
