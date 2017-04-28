# from sklearn.model_selection import KFold
#
# kf = KFold(3, shuffle=False)
# # kf = KFold(3,shuffle=True)
#
# for train_indices, test_indices in kf:
#     print(train_indices, test_indices)

from sklearn.model_selection import KFold
import numpy as np

kf = KFold(3, shuffle=True)
for train_indices, test_indices in kf.split(np.array([[1,2,3], [1,2,3], [3,4,5]])):
    print(train_indices, test_indices)