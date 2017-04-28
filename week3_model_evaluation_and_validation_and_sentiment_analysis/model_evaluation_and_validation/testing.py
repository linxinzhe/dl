from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[]])
y = np.array([[]])

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25)
