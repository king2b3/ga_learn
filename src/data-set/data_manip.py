"""data_manpip.py
Developer: Bayley King
Date: 2-19-2022
Descrition: Data set manipulation file.
"""

################################## Imports ###################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##############################################################################

################################## Globals ###################################
df = pd.read_csv('data-set/full_set.csv')
##############################################################################

X = df.drop('target', axis=1)
Y = df['target']

#80/20 split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

np.save('data-set/x_train.npy', X_train)
np.save('data-set/y_train.npy', Y_train)
np.save('data-set/x_test.npy', X_test)
np.save('data-set/y_test.npy', Y_test)