from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from preprocess import X, y

# Spliting the data for test and train
X_train = X[:4]
y_train = y[:4]
X_test = X[6:]
y_test = y[6:]

# creating a RF classifier
clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# using metrics module for accuracy calculation

print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
