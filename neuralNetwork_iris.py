import numpy as np 
import random
import copy
import time
#from sklearn.metrics import accuracy_score

##Neural Network for the IRIS dataset
##Sort the local and global variables accordingly

#Stores the number of hidden layers
hidden_layers = []
#Stores the dimension of each layers
dimension_vector = []
#Stores the weight matrix for each layer
weight_vector = {}
#Learning rate
eta = 0.01

s_l = {}
x_l = {}
X = []
S = []

##Data structure initialization to store the values for the entire data-set:
eta = 0.01
weight_vector = {}
x_l = {}
s_l = {}
##Initialization of the data structures to store all the computed values for the data points
W = []
WW = []
W_updated_all = []
X_backprop_all = []
W_all = []
X_all = []
S_all = []
delta_l_all = []
theta_dash_l_all = []
Ein_all = []
G_all = []
W_new_all = []

##Number of iterations to be performed
N = 20

##Loading the entire IRIS dataset and generating the labels
#Input: filename e.g. 'iris.txt'
#Output: Data in the form of list of lists and the corresponding labels
def getIrisData(filename):
    a = []
    labels = []
    labels_name = []
    labels_gen = []
    with open(filename,'r')as f:
        for line in f:
            s = line.split(',')
            a.append(s)
    random.shuffle(a)
    for line in a:
        labels.append(line[4])
    for i in labels:
        labels_name.append(i.split('\n'))
    for i in labels_name:
        if i[0] == 'Iris-versicolor':
            labels_gen.append(0)
        if i[0] == 'Iris-virginica':
            labels_gen.append(1)
        if i[0] == 'Iris-setosa':
            labels_gen.append(2)
    #Generating input vector
    for i in a:
        i[0] = float(i[0])
        i[1] = float(i[1])
        i[2] = float(i[2])
        i[3] = float(i[3])
        del i[4]
        i.insert(0,1)
    return labels_gen,a

##Loading just half the data set for binary classification
#Here we load data for only Iris-setosa and Iris-versicolor
#Input: filename e.g. 'iris.txt'
#Output: Data in the form of list of lists and the corresponding labels
def getIrisDataBinary(filename):
    a = []
    labels = []
    labels_name = []
    labels_gen = []
    with open(filename,'r')as f:
        for line in f:
            s = line.split(',')
            a.append(s)
    a = a[:100]
    random.shuffle(a)
    for line in a:
        labels.append(line[4])
    for i in labels:
        labels_name.append(i.split('\n'))
    for i in labels_name:
        if i[0] == 'Iris-setosa':
            labels_gen.append(0)
        if i[0] == 'Iris-versicolor':
            labels_gen.append(1)
    for i in a:
        i[0] = float(i[0])
        i[1] = float(i[1])
        i[2] = float(i[2])
        i[3] = float(i[3])
        del i[4]
        i.insert(0,1)
    return labels_gen,a

##Function to split the data into training and testing
#Input: Training data and testing data percent specified by a number (e.g. test = 30, train = 30)
		#Complete data and corresponding labels
#Output: Data and labels split into training set and testing set.
def splitData(train,test,labels,data):
    idx_train = (train/100)*len(labels)
    idx_train = int(idx_train)
    train_labels = labels[:idx_train]
    train_data = data[:idx_train]
    test_labels = labels[idx_train:]
    test_data = data[idx_train:]
    return train_labels, train_data, test_labels, test_data

##Function to generate the neural network based on the user specifications for the neural 
##network architecture 
#Input: none
#Output: A dictionary of weight vector for the entire neural network based on the dimensions
		#of the layer and the number of nodes in each layer.
def genNeuralNet():
    layers = input("Enter the number of layers:")
    layers = int(layers)
    total_layers = layers + 1
    for i in range(total_layers):
        hidden_layers.append(i)
    for i in range(len(hidden_layers)):
        dim = input("Enter the dimension of the layer:")
        dim = int(dim)
        dim +=1
        dimension_vector.append(dim)
    dimension_vector[0] = ip
    print(dimension_vector)
    #Generating a weight vector based on the architecture of the neural network
    weights = layers
    for i in range(len(dimension_vector)-1):
        dim1 = dimension_vector[i]
        dim2 = dimension_vector[i+1]-1
        weights = np.zeros([dim1,dim2])
        for x in np.nditer(weights, op_flags = ['readwrite']):
                x[...] = round(np.random.normal(0,1),2)
        weight_vector[i] = weights
    return hidden_layers, dimension_vector, weight_vector

##Function to generate the weights for the neural network based on the architecture the weight 
##vector is generated considering the bias node in each layer.
#Input: Empty dictionary of weight vector and dimension of each layer
#Output: A dictionary of weights for each layer
def genWeights(dimension_vector, weight_vector):
    for i in range(len(dimension_vector)-1):
        dim1 = dimension_vector[i]
        dim2 = dimension_vector[i+1]-1
        weights = np.zeros([dim1,dim2])
        for x in np.nditer(weights, op_flags = ['readwrite']):
            x[...] = round(np.random.normal(0,1),2)
        weight_vector[i] = weights
    return weight_vector
w3 = copy.deepcopy(W_new_all)

##Function for the forward Propagation in a Neural Network
#Input: Weight vector, x_l containing the input datapoint and s_l
#Output: Updated x_l and s_l for the entire neural network
def forwardPropagation(weight_vector, x_l, s_l):
    w = len(weight_vector)
    for i in range(w):
        j = i+1
        if j < w:
            s_l[i] = np.matmul(np.transpose(weight_vector[i]),x_l[i])
            x_l[j] = np.insert(np.tanh(s_l[i]),0,1)
        elif j == w:
            s_l[i] = np.matmul(np.transpose(weight_vector[i]),x_l[i])
            x_l[j] = np.around(np.tanh(s_l[i]),decimals = 1)
    return x_l, s_l

##Function to update the parameters: weight vector and x_l for backward propagation
def updateParameters(weight_vector,x):
    weight_vector_updated = {}
    xl_backprop = {}
    for i in weight_vector:
        weight_vector_updated[i] = (np.delete(weight_vector[i],0,0))
    counter = len(x)
    for i in range(counter-1):
        xl_backprop[i] = np.delete(x[i],0)
    xl_backprop[counter-1] = x[counter-1]
    return weight_vector_updated, xl_backprop, counter

##Function to update the weights for backprop removing the bias term weights
#Input: Weight vector which is a dictionary of weights for the neural network and an empty
		#dictionary to store the updated weights
#Output: Updated dictionary with the weights corresponding to the bias term removed.
def updateWeightsBP(weight_vector, weight_vector_updated):
    for i in weight_vector:
        weight_vector_updated[i] = (np.delete(weight_vector[i],0,0))
    return weight_vector_updated
    
##Function to update the input vector X_L for backprop by removing the bias term
#Input: x which contains the updated parameters for each layer after forward propagation
#Output: xl_backprop whic contains the updated parameters for each layer minus the bias node params
def updateX(x, xl_backprop):
    counter = len(x)
    for i in range(counter-1):
        xl_backprop[i] = np.delete(x[i],0)
    xl_backprop[counter-1] = x[counter-1]
    return xl_backprop,counter

##Function to return index lists. This is a helper function and generated for the convenience of the user 
def indexing(xl_backprop,s):
    xl_index = []
    sl_index = []
    for key in xl_backprop.keys():
        xl_index.append(key)
    xl_index = list(reversed(xl_index))
    for key in s.keys():
        sl_index.append(key)
    sl_index = list(reversed(sl_index))
    return xl_index, sl_index

##Function which implements the backward propagation algorithm and computes the sensitivity vectors for each layer
def backPropagation(xl_backprop, xl_index, sl_index, counter, y, weight_vector_updated):
    theta_dash_sl = {}
    delta_l = {}
    for i in xl_index:
        theta_dash = np.multiply(np.subtract(1,xl_backprop[i]),xl_backprop[i])
        theta_dash_sl[i] = theta_dash
    delta_l[counter-1] = np.multiply(np.multiply(2,np.subtract(x[counter-1],y)), theta_dash_sl[counter-1])
    for i in sl_index:
        delta_l[i] = np.multiply(theta_dash_sl[i],np.matmul(weight_vector_updated[i],delta_l[i+1]))
    return delta_l, theta_dash_sl

##Function to compute Ein at a particular data point
def computeEin(delta_l, counter, y, Num):
    d = delta_l[counter-1]
    E_in = 0
    E_in = E_in + np.multiply((1/Num),np.square(np.subtract(d,y)))
    return E_in

##Function to compute the gradient at each data-point
def computeGradient(delta_l, sl_index, xl_backprop):
    g = {}
    for i in range(len(sl_index)):
        j = i+1
        g[i] = np.matmul(xl_backprop[i],np.transpose(delta_l[i]))
    return g

##Function to update the weights accordingly
def updateWeights(weight_vector,g,eta):
    weight_vector_new = {}
    for i in range(len(weight_vector)):
        weight_vector_new[i] = np.subtract(weight_vector[i],np.multiply(eta,g[i]))
    return weight_vector_new

if __name__ == '__main__':

	print("Neural Network for Binary classification in the IRIS data set")
	
	##Testing the nueral network
	#Reading the data for binary classification
	gen_labels, gen_input = getIrisDataBinary('iris.txt')
	print("Data has been read!")

	#Splitting the data and labels into 70% training and 30% testing
	train = 70
	test = 30
	training_labels, training_data, testing_labels, testing_data = splitData(train,test,gen_labels,gen_input)
	print("Data has been split into training and testing data")
	
	ip = len(training_data[0])
	layers, dimensions, weight_parameter = genNeuralNet()
	for i in training_data:
	    a = copy.deepcopy(weight_vector)
	    weight_vector_new = genWeights(dimension_vector,a)
	    W_all.append(weight_vector_new)

	print("Weight vector has been generated!")    

	#print("W_all_first:",W_all)

	##The stopping criteria is the maximum number of iterations
	for j in range(N):
		w2 = copy.deepcopy(W_updated_all)
		w3 = copy.deepcopy(W_new_all)
		x1 = copy.deepcopy(X_all)
		x2 = copy.deepcopy(X_backprop_all)
		s1 = copy.deepcopy(S_all)
		d1 = copy.deepcopy(delta_l_all)
		t1 = copy.deepcopy(theta_dash_l_all)
		g1 = copy.deepcopy(G_all)
		e1 = copy.deepcopy(Ein_all)

	##Calculating X_l and S_l for all the input data points and storing it in the X_all, S_all lists
		for i in range(len(training_data)):
		    b = copy.deepcopy(x_l)
		    c = copy.deepcopy(s_l)
		    b[0] = training_data[i]
		    weight_current = W_all[i]
		    x = copy.deepcopy(x_l)
		    s = copy.deepcopy(s_l)
		    x,s = forwardPropagation(weight_current,b,c)
		    x1.append(x)
		    s1.append(s)
		
		##Updating parameters and back propagation
		for i in range(len(training_data)):
		    d = copy.deepcopy(weight_vector)
		    e = copy.deepcopy(weight_vector)
		    e = W_all[i]
		    weight_vector_updated = updateWeightsBP(e,d)
		    w2.append(weight_vector_updated)
		##Weight vector updated for backpropagation
		
		for i in range(len(training_data)):
		    f = copy.deepcopy(x_l)
		    g = copy.deepcopy(x_l)
		    f = x1[i]
		    xl_backprop, counter = updateX(f,g)
		    x2.append(xl_backprop)
		##Updated X_all for back propagation 
 
		xl_b = x2[0]
		s = s1[0]
		xl_index, sl_index = indexing(xl_b,s)
		
		##Backward Propagation implementation
		for i in range(len(training_data)):
		    y = training_labels[i]
		    h = copy.deepcopy(x_l)
		    xl_backprop = copy.deepcopy(x_l)
		    j = copy.deepcopy(weight_vector)
		    weight_vector_updated = copy.deepcopy(weight_vector)
		    xl_backprop =  x2[i]
		    weight_vector_updated = w2[i]
		    h,j = backPropagation(xl_backprop, xl_index, sl_index, counter, y, weight_vector_updated)
		    d1.append(h)
		    t1.append(j)
		##Backward Propagation implemented

		Num = len(training_data)
		##Computing error, calculating the gradient and updating the weights for the next iteration
		##Computing the error for the entire data set:
		for i in range(len(training_data)):
		    y = training_labels[i]
		    k = copy.deepcopy(x_l)
		    k = d1[i]
		    E_in = computeEin(k,counter,y,Num)
		    e1.append(E_in)

		##Computing the gradient for the entire dataset:
		for i in range(len(training_data)):
		    l = copy.deepcopy(x_l)
		    m = copy.deepcopy(x_l)
		    l = d1[i]
		    m = x2[i]
		    g = computeGradient(l,sl_index,m)
		    g1.append(g)

		##Updating the weights:
		for i in range(len(training_data)):
			n = copy.deepcopy(weight_vector)
			p = copy.deepcopy(x_l)
			n = W_all[i]
			p = g1[i]
			weight_vector_new = updateWeights(n,p,eta)
			w3.append(weight_vector_new)
		W_all = copy.deepcopy(W)

		W_all = w3
		
	train_error = e1[Num-1]
	print("Training error:", train_error)

	##Testing the data for test error
	for i in range(len(testing_data)):
		b = copy.deepcopy(x_l)
		c = copy.deepcopy(s_l)
		b[0] = testing_data[i]
		weight_current = W_all[i]
		x = copy.deepcopy(x_l)
		s = copy.deepcopy(s_l)
		x,s = forwardPropagation(weight_current,b,c)
		X_all.append(x)
		S_all.append(s)

	##Updating parameters and back propagation
	for i in range(len(testing_data)):
	    d = copy.deepcopy(weight_vector)
	    e = copy.deepcopy(weight_vector)
	    e = W_all[i]
	    weight_vector_updated = updateWeightsBP(e,d)
	    W_updated_all.append(weight_vector_updated)
	
	for i in range(len(testing_data)):
	    f = copy.deepcopy(x_l)
	    g = copy.deepcopy(x_l)
	    f = X_all[i]
	    xl_backprop, counter = updateX(f,g)
	    X_backprop_all.append(xl_backprop)
		##Updated X_all for back propagation
	
	#print(X_backprop_all)
	X_label = []
	for i in X_backprop_all:
		X_label.append(i[counter-1])


	xl_b = X_backprop_all[0]
	s = S_all[0]
	xl_index, sl_index = indexing(xl_b,s)

##Backward Propagation implementation
	for i in range(len(testing_data)):
	    y = testing_labels[i]
	    h = copy.deepcopy(x_l)
	    xl_backprop1 = copy.deepcopy(x_l)
	    j = copy.deepcopy(weight_vector)
	    weight_vector_updated = copy.deepcopy(weight_vector)
	    xl_backprop1 =  X_backprop_all[i]
	    weight_vector_updated = W_updated_all[i]
	    h,j = backPropagation(xl_backprop1, xl_index, sl_index, counter, y, weight_vector_updated)
	    delta_l_all.append(h)
	    theta_dash_l_all.append(j)
		##Backward Propagation implemented

	Num1 = len(testing_data)
	##Computing error, calculating the gradient and updating the weights for the next iteration
	##Computing the error for the entire data set:
	for i in range(len(testing_data)):
	    y = testing_labels[i]
	    k = copy.deepcopy(x_l)
	    k = delta_l_all[i]
	    E_in = computeEin(k,counter,y,Num1)
	    Ein_all.append(E_in)
	print("Testing error:",Ein_all[Num1-1])

	# print("predicted:", X_label)
	# sign_list = []
	# for i in X_label:
	# 	sign_list.append(np.sign(i))

	# print("signs:", sign_list)
	# print("truth:", testing_labels)
	# predicted_list = []
	# for i in range(len(sign_list)):
	# 	if sign_list[i] == -1:
	# 		sign_list[i] = 0
	# 	elif sign_list[i] == 1:
	# 		sign_list[i] = 1

	# print("predicted:", sign_list)



