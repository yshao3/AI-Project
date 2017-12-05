# Save Model Using Pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

loaded_model = pickle.load(open("A30.sav", 'rb'))
result = loaded_model.predict(np.array([[40]*30]))
print(result)