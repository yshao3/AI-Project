from sklearn.neural_network import MLPRegressor, MLPClassifier
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_score = []
test_score = []
label_score = []
def getdata(path, period):
	X_train = []
	Y_train = []
	Y_Label = []
	data = []
	count = 0
	x = []
	prev = 0
	with open(path,'rb') as p:
		reader = csv.reader(p)
		reader.next()
		for row in reader:
			count += 1
			tmp = float (row[1])
			if tmp - prev > 0: label = 1
			else: label = 0
			if count > period:
				data.append([x,[tmp, label]])
				# X_train.append(x)
				# Y_train.append(tmp)
				x = x[1:] + [tmp]
				# x = [tmp]
				# count = 1
			else: x = x + [tmp]
			prev = tmp
			# print(len(x), period, tmp)
	sorted(data)
	for i in data:
		X_train.append(i[0])
		Y_train.append(i[1][0])
		Y_Label.append(i[1][1])
	return X_train,Y_train,Y_Label

def train(filename,X_train, Y_train, Y_Label):

	nn = MLPRegressor(hidden_layer_sizes= (100,100,100))
	X = np.array(X_train[0:1000])
	Y = np.array(Y_train[0:1000])
	nn.fit(X, Y)
	test = nn.predict(np.array(X_train[1000:]))
	# plot predict data and real data
	train = np.array(Y_train[1000:])
	x = np.arange(0,len(X_train)-1000)
	plt.plot(x,train)
	plt.plot(x,test)
	plt.show()
	print (Y_train[1000:])

	train_pred = nn.score(np.array(X_train[0:1000]), np.array(Y_train[0:1000]), sample_weight=None)
	test_pred = nn.score(np.array(X_train[1000:]), np.array(Y_train[1000:]), sample_weight=None)
	print test_pred
	# predict label using trained regression
	# label_pred = test.tolist()
	# for i in range(len(label_pred)-1, 0,-1):
	# 	if label_pred[i]-label_pred[i-1] > 0: label_pred[i] = 1
	# 	else: label_pred[i] = 0
	# label_pred[0] = 0
	# count = 0
	# for (i, j) in zip(label_pred,Y_Label[1000:]):
	# 	if label_pred[i] == Y_Label[i]: count+=1
	# label_score.append(count*1.0/len(label_pred))


	pickle.dump(nn, open(filename, 'wb'))
	print (nn.get_params())

	# NNClassifier
	# nn1 = MLPClassifier(max_iter = 200000)
	# # nn = Regressor(
	# # layers =[ Layer("Sigmoid", units=30), Layer("Sigmoid", units=10), Layer("Sigmoid", units=3), Layer("Linear")],learning_rate=0.02,n_iter=100)
	# # tmp = norm(X_train[0:2000])
	# X = np.array(X_train[0:1000])
	# Y = np.array(Y_Label[0:1000])
	
	# # xp = []
	# # for i in X_train: 
	# # 	xp += [i[0]]
	# nn1.fit(X, Y)
	# print nn1.predict(np.array(X_train[1000:]))
	# # print (Y_Label[1000:])
	# print nn1.score(np.array(X_train[0:1000]), np.array(Y_Label[0:1000]), sample_weight=None)
	# print nn1.score(np.array(X_train[1000:]), np.array(Y_Label[1000:]), sample_weight=None)
	# print (nn1.get_params())

	#   linear regression result
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

def main():

	# plot scores according different number of features
	# for i in range(5,40):
	# 	X_train, Y_train, Y_Label  =getdata(path, i)
	# 	train(filename, X_train, Y_train, Y_Label)
	# x = np.arange(5,40)
	# plt.plot(x,label_score)
	# plt.show()
	X_train, Y_train, Y_Label  =getdata(path, 30)
	train(filename, X_train, Y_train, Y_Label)

if __name__=='__main__':
	if len(sys.argv) < 2:
		print "error: parameters are not enough"
	path = sys.argv[1]
	filename = sys.argv[2]

	main()


