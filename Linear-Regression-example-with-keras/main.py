import numpy as np
import argparse
import csv

# import matplotlib.pyplot as plt
''' 
You are only required to fill the following functions
mean_squared_loss
mean_squared_gradient
mean_absolute_loss
mean_absolute_gradient
mean_log_cosh_loss
mean_log_cosh_gradient
root_mean_squared_loss
root_mean_squared_gradient
preprocess_dataset
main

Don't modify any other functions or commandline arguments because autograder will be used
Don't modify function declaration (arguments)

'''

def mean_squared_loss(xdata, ydata, weights):

	guess = np.dot(xdata,weights)
	samples = np.shape(guess)[0]
	err = (0.5/samples)*np.sum(np.square(ydata-guess))
	return err
	raise NotImplementedError

def mean_squared_gradient(xdata, ydata, weights):

	samples = np.shape(xdata)[0]
	guess = np.dot(xdata,weights)
	gradient = (1/samples)*np.dot(xdata.T,(guess-ydata))
	return gradient

	raise NotImplementedError

def mean_absolute_loss(xdata, ydata, weights):

	guess = np.dot(xdata,weights)
	samples = np.shape(guess)[0]
	err = 0.5*samples*np.sum(np.absolute(ydata-guess))
	return err
	raise NotImplementedError

def mean_absolute_gradient(xdata, ydata, weights):

	samples = np.shape(xdata)[0]
	guess = np.dot(xdata,weights)
	gradient = (1/samples)*np.dot(xdata.T,(guess-ydata))
	return gradient

	raise NotImplementedError

def mean_log_cosh_loss(xdata, ydata, weights):

	guess = np.dot(xdata,weights)
	samples = np.shape(guess)[0]
	err = samples*np.sum(np.log(np.cosh(ydata-guess)))
	return err
	raise NotImplementedError

def mean_log_cosh_gradient(xdata, ydata, weights):

	guess = np.dot(xdata,weights)
	simplerr = np.multiply(2,ydata-guess)
	samples = np.shape(guess)[0]
	derivative = np.divide(np.exp(simplerr)-1,np.exp(simplerr)+1)
	gradient = (1/samples)*np.dot(xdata.T,derivative)
	return gradient

	raise NotImplementedError

def root_mean_squared_loss(xdata, ydata, weights):

	guess = np.dot(xdata,weights)
	samples = np.shape(guess)[0]
	err = np.sqrt(np.divide(np.sum(np.square(ydata.T-guess)),samples))
	return err
	raise NotImplementedError

def root_mean_squared_gradient(xdata, ydata, weights):

	samples = np.shape(xdata)[0]
	gradient = -weights.T/np.sqrt(samples)
	return gradient

	raise NotImplementedError

class LinearRegressor:

	def __init__(self, dims):
		
		self.dims = dims
		self.W = np.random.rand(dims)
		#self.W = np.random.uniform(low=0.0, high=1.0, size=dims)
		return

		raise NotImplementedError

	def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=100, lr=1):
		errlog = []
		samples = np.shape(xtrain)[0]
		for iterations in range(epoch):
			self.W = self.W - lr*gradient_function(xtrain,ytrain,self.W)
			errlog.append(loss_function(xtrain,ytrain,self.W))
		return errlog
		raise NotImplementedError

	def predict(self, xtest):
		
		# This returns your prediction on xtest
		return np.dot(xtest,self.W)
		raise NotImplementedError


def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []

	with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)

	return np.array(xtrain), np.array(ytrain), np.array(xtest)

def one_hot_encoding(value_list, classes):
    res = np.eye(classes)[value_list.reshape(-1)]
    return res.reshape(list(value_list.shape)+[classes])

norm_dict = {}

dictionary_of_classes_for_features = {
	2 : 5,
	3 : 25,
	5: 8,
	7: 5
}

dictionary_of_days = {
	'Monday' : 1,
	'Tuesday': 2,
	'Wednesday': 3,
	'Thursday' : 4,
	'Friday' : 5,
	'Saturday': 6,
	'Sunday' : 7
}

def slicer(arr, beg, end):
	return np.array([i[beg:end] for i in arr]).reshape(-1, 1)
"""	
#for normalization of parametes 'wind speed' and 'humidity' uncoment
def normalize(arr):
	arr = arr
	if not norm_dict: # make dictionary once at training to be used later during test
		# for i in range(arr.shape[1]):
		norm_dict['init'] = [np.min(arr), np.max(arr)]
		#norm_dict['init'] = [np.mean(arr), np.std(arr)]
	# for i in range(arr.shape[1]):
	arr = np.array([(x - norm_dict['init'][0])/(norm_dict['init'][1] - norm_dict['init'][0]) for x in arr]) # min-max
	#arr = np.array([(x - norm_dict['init'][0])/(norm_dict['init'][1]) for x in arr]) # standardization
		
	return arr
"""
# 4 hours band
# 1/-1 encoding
# use feature selection and tuning in Jupyter then apply it back here

def preprocess_dataset(xdata, ydata=None):
	
	# converting weekdays to numeric for one_hot_encoding
    """

	#for normalization of parametes 'wind speed' and 'humidity' uncoment
	xdata[:, 10] = normalize(xdata[:, 10].astype('float'))# normalized
	xdata[:, 11] = normalize(xdata[:, 10].astype('float'))"""
    xdata[:, 5] = [dictionary_of_days[i] for i in xdata[:, 5]]

    cat_cols = [2, 3, 5, 7]

	
    for i in cat_cols:
		# dropping 2 columns for C-1 encoding and removing additional 0 column
        t = one_hot_encoding(xdata[:, i].astype('int'), dictionary_of_classes_for_features[i])[:, 2:]
        xdata = np.concatenate((xdata, t),axis=1)
	
    xdata = np.delete(xdata, cat_cols, 1) # removing useless columns
    xdata = np.delete(xdata, 6, 1)
    xdata = np.delete(xdata, 8, 1)
	
    # extracting features from date
    month = slicer(xdata[:, 1], 5,7)
    t = one_hot_encoding(month[:,0].astype('int'), 13)[:, 2:]
    xdata = np.concatenate((xdata, t), axis=1)
    date = slicer(xdata[:, 1], 8, 10)
    week = np.ceil(date.astype('int') / 7)  # week of month
    t = one_hot_encoding(week[:,0].astype('int'), 6)[:, 2:]
    xdata = np.concatenate((xdata, t), axis=1)


    xdata = xdata[:,2:] # dropping first 2 unnecessary columns
    print(xdata[0:5])
	
    xdata = xdata.astype('float32')
    bias = np.ones((np.shape(xdata)[0],1))
    xdata = np.concatenate((bias,xdata),axis=1)

    if ydata is None:
        return xdata
    ydata = ydata.astype('float32')
    return xdata,ydata
    raise NotImplementedError

dictionary_of_losses = {
	'mse':(mean_squared_loss, mean_squared_gradient),
	'mae':(mean_absolute_loss, mean_absolute_gradient),
	'rmse':(root_mean_squared_loss, root_mean_squared_gradient),
	'logcosh':(mean_log_cosh_loss, mean_log_cosh_gradient),
}

"""
#For outliers removal from wind speed column uncomment
def out(x, std, mean):
    if ((x < mean + 2 * std)and (x > mean - 2 * std)):
        return 0
    else:
        return 1


def outlier(xtrain, ytrain, std, mean):
    a =[]
    for i in xtrain[:, 11].astype('float32'):
        a.append(out(i,std, mean))
    a = np.array(a)
    xdata = np.concatenate((xtrain, a.reshape(-1, 1)), axis=1)
    ytrain = np.delete(ytrain, np.argwhere(xdata[:, -1].astype('int') > 0), 0)
    xdata = np.delete(xdata, np.argwhere(xdata[:, -1].astype('int') > 0), 0)
    xdata = np.delete(xdata, -1, 1)
    return (xdata, ytrain)"""

def main():
    # You are free to modify the main function as per your requirements.
	# Uncomment the below lines and pass the appropriate value

    xtrain, ytrain, xtest = read_dataset(args.train_file, args.test_file)

    """
    #For outliers removal from wind speed column uncomment
    std = np.std(xtrain[:, 11].astype('float32'))
    mean = np.mean(xtrain[:, 11].astype('float32'))
    xtrain, ytrain =outlier(xtrain, ytrain, std, mean)"""
    xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
    xtestprocessed = preprocess_dataset(xtest)
	
    model = LinearRegressor(np.shape(xtrainprocessed)[1])

    # The loss function is provided by command line argument
    loss_fn, loss_grad = dictionary_of_losses[args.loss]

    errlog = model.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, args.epoch, args.lr)
    ytest = model.predict(xtestprocessed)
    ytest = ytest.astype('int')
    output = [(i,np.absolute(ytest[i])) for i in range(len(ytest))]
    np.savetxt("output.csv",output,delimiter=',',fmt="%d",header="instance (id),count",comments='')
    np.savetxt("error.log",errlog,delimiter='\n',fmt="%f")


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()

	main()
