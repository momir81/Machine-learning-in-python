import sys
import PIL.Image
import scipy.misc, scipy.optimize, scipy.io,scipy.special
from numpy import *

import pylab

from matplotlib import pyplot,cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab




def part1():
    print 'Loading and Visualizing data...\n'

    data=scipy.io.loadmat('ex3data1.mat')
    X,y = data['X'],data['y']#X is 5000x400, y is 5000x1 
    m,n = shape(X)
    input_layer_size = 400
    num_labels = 10
    lamda = 0.1
    #theta = zeros((1,n))
    displayData(X)

    
    theta = oneVsAll( X, y, num_labels, lamda )
    predictOneVsAll( theta, X, y )

    displayData(X,theta)
    
def displayData(X, theta=None):
    width = 20
    rows, cols = 10,10#100 slika (10 horizontalno i vert), svaka slika je 20x20 piksela 
    out = zeros((width*rows,width*cols))#izlaz je 200x200 slika 

    rand_indices = random.permutation(5000)[0:rows*cols]# randomly bira 10x10 brojeva iz opsega od 5000, niz od sto elemenata
    
    counter = 0

    for y in range(0,rows):
        for x in range(0,cols):
            start_x = x*width
            start_y = y*width
            out[start_x:start_x+width,start_y:start_y+width] = X[rand_indices[counter]].reshape(width,width).T
            counter +=1
            """
            svaka slika je smestena na 20 piksela udaljenosti od prethodne
            po horizontali i vertikali, jer je svaka slika velicine 20x20 i hocu
            10 takvih slika po horizontali i verticali sto daje ukupno 100 slika
            gde je svaka slika 20x20 piksela
            """

    img = scipy.misc.toimage(out)
    figure = pyplot.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    if theta is not None:
        result_matrix = []
        X_biased = c_[ones(shape(X)[0]),X]
        #each class is labeled from 1 to 10 and this code calculates for each field in 10x10
        #matrix probalibilty of specific class
        for idx in rand_indices:
            result = (argmax(theta.T.dot(X_biased[idx])) +1) % 10
            result_matrix.append(result)
        result_matrix = array(result_matrix).reshape(rows,cols).transpose()

        print result_matrix
            
    pyplot.show()

def lrCostFunction(theta, X,y,lamda):
    #theta is 1x401 dim 401 jer se dodaje theta0
    J=0
    m = len(y)
    #grad = zeros(size(theta))

    h = 1/(1 + exp(-X.dot(theta.T)))

    y1 = log(h).T.dot(-y)

    y2 = log(1-h).T.dot(1-y)

    regul = lamda/(2*m)*sum(theta[1:]**2)


    J = (y1-y2)/m + regul

    #print 'cost function value', J
    
    return J

def gradientCost(theta,X,y,lamda):
    m = shape(X)[0] #m 5000 vector

    h = 1/(1 + exp(-X.dot(theta.T)))
    grad = (X.T.dot(h-y))/m
    grad[1:] += lamda/m * theta[1:]
    return grad

def oneVsAll(X,y,num_labels,lamda):
    m,n = shape(X)

    X = c_[ones((m, 1)), X]#add the first row of ones to matrix X,now dim is 5001x400

    all_theta = zeros((n+1,num_labels))#401X10

    for k in range(0, num_labels):
        theta = zeros((n+1,1)).reshape(-1)

        y_label = ((y==(k+1))+0).reshape(-1)

        result = scipy.optimize.fmin_cg(lrCostFunction,fprime=gradientCost,x0=theta,args=(X,y_label,lamda),maxiter=50,disp=False,full_output =True)

        all_theta[:,k] = result[0]

    print "%d Cost: %5f" % (k+1,result[1])
    return all_theta
    
def predictOneVsAll(theta,X,y):

    m,n = shape(X)
    X 	= c_[ones((m, 1)), X]

    correct = 0
    for i in range(0, m ):
        prediction  = argmax(theta.T.dot( X[i] )) + 1
        actual 	= y[i]# print "prediction = %d actual = %d" % (prediction, actual)
        if actual == prediction:
            correct += 1
    print "Accuracy: %.2f%%" % (correct * 100.0 / m )

def part2():
    data1 = scipy.io.loadmat('ex3data1.mat')
    X,y = data1['X'],data1['y']
    
    m = shape(X)[0]
    
    data=scipy.io.loadmat('ex3weights.mat')
    theta1,theta2 = data['Theta1'],data['Theta2']#theta1 25x401,theta2 10x26
    #print 'Celo X', X
    out = displayData2(X,theta1,theta2)
    correct = 0
    #rows,cols=20
    
    k,l=shape(out)#100x100
    width=k/5
    
    for i in range(0,5):
        for j in range(0,5):
            
                print 'i',i
                print 'j',j
                start_x = i * 20
                start_y = j * 20
                #out[start_x:start_x+width, start_y: start_y+width]
                #pred=predict(out[start_x:start_x +width, start_y: start_y +width],theta1,theta2)
                displayData3(out[start_x:start_x +width, start_y: start_y +width])
                #print "\nNeural Network Prediction: %d (digit %d)\n" % (pred,mod(pred,10))
                
    #ne prikazuje istu cifru kao u prethodnoj slici jer displayData2 uzima random
    #print 'X 400', X[[0],:]
    
def predict(X,theta1,theta2):
    #print shape(X)
    
    a1 = r_[ones((1, 1)), X.reshape( shape(X)[0], 1 )]
    z2 = sigmoid( theta1.dot( a1 ))
    z2 = r_[ones((1, 1)), z2]
    z3 = sigmoid(theta2.dot( z2 ))
    return argmax(z3) + 1

def displayData3(X):
    out = X
    
    img = scipy.misc.toimage( out )
    figure = pyplot.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    pyplot.show()

def displayData2(X, theta1 = None, theta2 = None):
	m,n = shape(X)
            
	width = sqrt(n)
	rows, cols = 5, 5

	out = zeros((width*rows, width*cols))

	rand_indices = random.permutation(m)[0:rows * cols]

	counter = 0
	for y in range(0, rows):
		for x in range(0, cols):
			start_x = x * width
			start_y = y * width
			out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
			counter += 1

	img = scipy.misc.toimage( out )
	figure = pyplot.figure()
	axes = figure.add_subplot(111)
	axes.imshow(img)


	if theta1 is not None and theta2 is not None:
		result_matrix 	= []
		
		for idx in rand_indices:
			result = predict( X[idx], theta1, theta2 )
			result_matrix.append( result )

		result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
		print result_matrix

	pyplot.show( )
	
	return out#5 slika od po 20x20to je 100x100
	
def sigmoid(z):
   return scipy.special.expit(z)
	
def main():
    part1()
    part2()
    


if __name__ == '__main__':
	main()
