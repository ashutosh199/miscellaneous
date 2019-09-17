import numpy as np
import csv
def out(x):
    if ((x < mean + 2 * std)and (x > mean - 2 * std)):
        return 0
    else:
        return 1
xtrain = []
ytrain = []
trainfile = 'C:/Users/anand/Desktop/shreyansh sir/ashutosh/train.csv'
with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])
xtrain=np.array(xtrain)
ytrain=np.array(ytrain)
a=[]
#print(xtrain.shape)
std=np.std(xtrain[: , 11].astype('float32'))
mean=np.mean(xtrain[: , 11].astype('float32'))
#print(std)
count=0
for i in xtrain[:, 11].astype('float32'):
    count+=1
    #print(count)
    a.append(out(i))
a=np.array(a)
#print(a.shape)
xdata = np.concatenate((xtrain, a.reshape(-1, 1)), axis=1)
ytrain=np.delete(ytrain,np.argwhere(xdata[:, -1].astype('int') > 0), 0)
xdata = np.delete(xdata, np.argwhere(xdata[:, -1].astype('int') > 0), 0)
print(xdata.shape)
xdata = np.delete(xdata, -1, 1)
xdata = np.concatenate((xdata, ytrain.reshape(-1,1)), axis=1)


print(xdata.shape)
print(ytrain.shape)


