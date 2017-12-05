from sknn.mlp import Regressor, Layer, Classifier
from sklearn.neural_network import MLPRegressor
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train = []
Y_train = []
nn = None
def getdata(path, period):
	data = []
	count = 0
	x = []
	with open(path,'rb') as p:
		reader = csv.reader(p)
		reader.next()
		for row in reader:
			count += 1
			tmp = float (row[4])
			if count > period:
				data.append([x,[tmp]])
				# X_train.append(x)
				# Y_train.append(tmp)
				x = x[1:] + [tmp]
				# x = [tmp]
				# count = 1
			else: x = x + [tmp]
			# print(len(x), period, tmp)
	sorted(data)
	for i in data:
		X_train.append(i[0])
		Y_train.append(i[1])
	print(len(X_train))

def train(filename):
	nn = MLPRegressor()
	# nn = Regressor(
	# layers =[ Layer("Sigmoid", units=30), Layer("Sigmoid", units=10), Layer("Sigmoid", units=3), Layer("Linear")],learning_rate=0.02,n_iter=100)
	# tmp = norm(X_train[0:2000])
	X = np.array(X_train[0:1000])
	Y = np.array(Y_train[0:1000])
	# xp = []
	# for i in X_train: 
	# 	xp += [i[0]]
	nn.fit(X, Y)
	print nn.predict(np.array(X_train[1000:]))
	print (Y_train[1000:])
	print nn.score(np.array(X_train[0:1000]), np.array(Y_train[0:1000]), sample_weight=None)
	print nn.score(np.array(X_train[1000:]), np.array(Y_train[1000:]), sample_weight=None)
	pickle.dump(nn, open(filename, 'wb'))
	print (nn.get_params())
# 	lr = LinearRegression()
# 	lr.fit(X, Y)
# 	filename = 'LRmodel.sav'
# 	pickle.dump(lr, open(filename, 'wb'))
# 	diabetes_y_pred = lr.predict(np.array(X_train[2000:]))
# 	for i in range(2000, len(Y)):
# 		print (Y_train[i],diabetes_y_pred[i-2000])

# 	# The coefficients
# 	print('Coefficients: \n', lr.coef_)
# 	# The mean squared error
# 	print("Mean squared error: %.2f" % mean_squared_error(np.array(Y_train[2000:]), diabetes_y_pred))
# 	print('Variance score: %.2f' % r2_score(np.array(Y_train[2000:]), diabetes_y_pred))

# # Plot outputs
# 	print(lr.get_params)
# 	plt.scatter(xp[2000:], Y_train[2000:],  color='black')
# 	plt.plot(xp[2000:], diabetes_y_pred, color='blue', linewidth=3)
	
# 	plt.xticks(())
# 	plt.yticks(())

# 	plt.show()
	# for i in range(2000, len(X_train)):
	# 	Y_pre = nn.predict(np.array(norm([X_train[i]])))
	# 	Y_p = Y_pre.tolist()
	# 	print(Y_train[i], Y_p[0])
def norm(x):
	tmp = []
	ma = [0]*len(x[0])
	mi = [0]*len(x[0])
	for i in x:
		for j in range(0, len(i)):
			ma[j] = max(ma[j], i[j])
			mi[j] = min(mi[j], i[j])
	for i in x:
		vec = []
		for j in range(0, len(i)):
			vec.append((i[j]-mi[j])/(ma[j]-mi[j]))
		tmp.append(vec)
	return tmp

# getdata("individual_stocks_5yr/A_data.csv", 5)
def main():
	# start_date = date(2016,3,6)
	# end_date = date(2016,5,12)
	getdata(path, int (period))
	train(filename)
	# print(X_train, Y_train)
if __name__=='__main__':
	if len(sys.argv) < 2:
		print "error: parameters are not enough"
	path = sys.argv[1]
	period = sys.argv[2]
	filename = sys.argv[3]

	main()



# nn = Classifier(
# 	layers =[ Layer("Rectifier", units=100),
#         Layer("Linear")],
#     learning_rate=0.02,
#     n_iter=10)
# nn.fit(X_train, Y_train)
# y_valid = nn.predict(x_valid)