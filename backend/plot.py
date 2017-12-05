import pickle

filename = 'lrmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model.get_parameters())
print(loaded_model)