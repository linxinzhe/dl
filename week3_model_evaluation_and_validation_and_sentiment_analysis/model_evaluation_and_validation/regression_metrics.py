from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[]])
y = np.array([[]])
classifier = LinearRegression()
classifier.fit(X, y)
guesses = classifier.predict(X)
error = mean_absolute_error(y,
                            guesses)  # but has a problem that it is not differentiate which can't implemented gradient descent method
error = mean_squared_error(y,
                           guesses)  # so use mean_squared_error

# --- R2 Score
from sklearn.metrics import r2_score

y_true = [1, 2, 4]
y_pred = [1.3, 2.5, 3.7]

print(r2_score(y_true, y_pred))
