
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
import csv
import sys
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from random import shuffle
import pandas as pd
import seaborn as sns

# params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
# 	           'learning_rate_init': 0.2},
# 	          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
# 	           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
# 	          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
# 	           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
# 	          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
# 	           'learning_rate_init': 0.2},
# 	          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
# 	           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
# 	          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
# 	           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
# 	          {'solver': 'adam', 'learning_rate_init': 0.01}]

# labels = ["constant learning-rate", "constant with momentum",
# 	          "constant with Nesterov's momentum",
# 	          "inv-scaling learning-rate", "inv-scaling with momentum",
# 	          "inv-scaling with Nesterov's momentum", "adam"]

# plot_args = [{'c': 'red', 'linestyle': '-'},
# 	             {'c': 'green', 'linestyle': '-'},
# 	             {'c': 'blue', 'linestyle': '-'},
# 	             {'c': 'red', 'linestyle': '--'},
# 	             {'c': 'green', 'linestyle': '--'},
# 	             {'c': 'blue', 'linestyle': '--'},
# 	             {'c': 'black', 'linestyle': '-'}]
data ={}
minimum = 0
maximum = 0
nn = None
def getdata(path):
	mi = 1000000000
	ma = 0
	with open(path,'rb') as p:
		reader = csv.reader(p)
		reader.next()
		for row in reader:
			tmp = []
			key = row[0]
			if int((key.split("-"))[0]) >17 : key = (1900+int((key.split("-"))[0]))*12 +int((key.split("-"))[1])
			else: key = (2000+int((key.split("-"))[0]))*12+int((key.split("-"))[1])
			if key < mi: 
				mi = key
			if key > ma: 
				ma = key
			data[key] = [float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10])]
	return (mi,ma)


def gettrain(feature):
	X_train = []
	Y_train = []
	Y_Label = []
	for i in range(minimum, maximum+1):
		if i not in data or data[i][0] == 0: continue
		tmp = []
		flag = True
		if i-1 not in data or data[i-1][0] == 0: 
			label = -1
		elif data[i][0] > data[i-1][0]: label = 1
		else: label = 0
		for f in feature:
			if data[i][f] == 0: 
				flag = False
				continue
			tmp.append(data[i][f])
		if flag: 
			X_train.append(tmp)
			Y_train.append(data[i][0])
			Y_Label.append(label)
	return X_train, Y_train, Y_Label

	# print X_train
def train(filename, a, b, X_train, Y_train,Y_Label):
	m = len(X_train)*a/b

	# X_train = norm(X_train)

	nn = MLPRegressor()
	# shuffle(zip(X_train, Y_train))
	X = np.array(X_train[0:m])
	Y = np.array(Y_train[0:m])

	nn.fit(X, Y)
	test = nn.predict(np.array(X_train[m:]))
	# plot predict data and real data
	train = np.array(Y_train[m:])
	x = np.arange(0,len(Y_train[m:]))
	plt.plot(x,train)
	plt.plot(x,test)
	plt.show()
	print nn.score(np.array(X_train[0:m]), np.array(Y_train[0:m]), sample_weight=None)
	print nn.score(np.array(X_train[m:]), np.array(Y_train[m:]), sample_weight=None)
	pickle.dump(nn, open(filename, 'wb'))
	print (nn.get_params())

	X_class = []
	Y_class = []
	for i in range(0, len(X_train)):
		if Y_Label[i] == -1: continue
		X_class.append(X_train[i])
		Y_class.append(Y_Label[i])
	m = len(X_class)*a/b
	X_class = norm(X_class)
	nn1 = MLPClassifier(hidden_layer_sizes=(50,50,50,),max_iter=20000, learning_rate_init=0.02)
	nn2 = SVC()
	X = np.array(X_class[0:m])
	Y = np.array(Y_class[0:m])
	nn1.fit(X, Y)
	nn2.fit(X, Y)
	print nn1.predict(np.array(X_class[:m]))
	print nn2.predict(np.array(X_class[:m]))
	print (Y_class[:m])
	print nn1.score(np.array(X_class[0:m]), np.array(Y_class[0:m]), sample_weight=None)
	print nn1.score(np.array(X_class[m:]), np.array(Y_class[m:]), sample_weight=None)
	print nn2.score(np.array(X_class[0:m]), np.array(Y_class[0:m]), sample_weight=None)
	print nn2.score(np.array(X_class[m:]), np.array(Y_class[m:]), sample_weight=None)
	y_score = nn2.fit(np.array(X_class[0:m]), np.array(Y_class[0:m])).decision_function(X_class[m:])
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	fpr, tpr, _ = roc_curve(np.array(Y_class[m:]), y_score)
	roc_auc = auc(fpr, tpr)

	# Compute micro-average ROC curve and ROC area
	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr,
 #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# # plt.xlim([0.0, 1.0])
	# # plt.ylim([0.0, 1.05])
	# # plt.xlabel('False Positive Rate')
	# # plt.ylabel('True Positive Rate')
	# # plt.title('Receiver operating characteristic example')
	# # plt.legend(loc="lower right")
	# # plt.show()
	# print (nn1.get_params())
	# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	# for ax in axes.ravel():plot_on_dataset(X, Y, ax=ax, name="features")

	# fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
	# plt.show()



def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)



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
# plt.figure()
minimum, maximum = getdata("features.csv")
#all features
X,Y,Z = gettrain([2,3,5,7,8,9])
train("features.sav", 3,4, X,Y,Z)
#2 features
X ,Y,Z= gettrain([2,3])
train("features.sav", 3,4, X,Y,Z)
#3 features
X,Y,Z = gettrain([2,3,5])
train("features.sav", 3,4, X,Y,Z)
#4 features
X,Y,Z = gettrain([2,3,5,7])
train("features.sav", 3,4, X,Y,Z)
#5 features
X,Y,Z = gettrain([2,3,5,7,8])
train("features.sav", 3,4, X,Y,Z)

# plot ROC curve
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('SVM model ROC curve different number of features')
# plt.legend(loc="lower right")
# plt.show()

# plot feature correlation
# test = np.array(X_train)
# d = pd.DataFrame({'PE':test[:,0],
# 	'DJ':(np.array(X_train))[:,1],
# 	'DJn':(np.array(X_train))[:,2],
# 	'Gold':(np.array(X_train))[:,3],
# 	'Goldn':(np.array(X_train))[:,4],
# 	'Oil':(np.array(X_train))[:,5],
# 	'Oiln':(np.array(X_train))[:,6],
# 	'Libor':(np.array(X_train))[:,7]}

#                 )
# corr = d.corr()
# print corr
# plt.matshow(corr)
# plt.title('features correlation', y=1.1)
# plt.show()
