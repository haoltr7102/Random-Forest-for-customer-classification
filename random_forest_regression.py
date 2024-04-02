import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values
    #print(X)
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X = SC.fit_transform(X)

# split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 0)

# visualize
from matplotlib.colors import ListedColormap
def VisualizingDataset(X_, Y_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    for i, label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_  == label], X2[Y_ == label], color = ListedColormap(("red", "green"))(i), label = label)
       
VisualizingDataset(X, Y)
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
#plt.show()

### random forest classification
# training
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion ='entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# visualization
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, classifier.predict(X_train))
print("Train: \n", cm)

# visualizing result
def VisualizingResult(model, X_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    X1_range = np.arange(start = X1.min()-1, stop = X1.max()+1, step = 0.01)
    X2_range = np.arange(start = X2.min()-1, stop = X2.max()+1, step = 0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
    X_grid = np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid = model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5, cmap = ListedColormap(("red", "green")))

VisualizingResult(classifier, X_train)
VisualizingDataset(X_train, Y_train)
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
#plt.show()

# test dataset
y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
print("\nTest: \n", cm)

# accuracy
from sklearn import metrics
print("\nAccuracy:", metrics.accuracy_score(Y_test, y_pred))



