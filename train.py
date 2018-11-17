# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn.externals import joblib
import os
import numpy as np
import pickle

os.makedirs('./outputs', exist_ok=True)

run = Run.get_submitted_run()

X, y = load_boston(return_X_y=True)

run = Run.get_submitted_run()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

alphas = np.arange(0.0, 1.0, 0.05)
for alpha in alphas:
    # Use Ridge algorithm to create a regression model
    reg = Ridge(alpha=alpha)
    reg.fit(data["train"]["X"], data["train"]["y"])

    model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    with open(model_file_name, "wb") as file:
        joblib.dump(value=reg, filename=os.path.join('./outputs/',
                                                     model_file_name))

