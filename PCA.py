import sys

from numpy import *

import scipy.misc,scipy.io,scipy.optimize

from matplotlib import pyplot, cm, colors, lines

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import RandomizedPCA

from K import *

def displayData(X):

    width = 32

    rows=cols = int(sqrt(shape(X)[0]))

    out = zeros((width*rows,width*cols))



    counter = 0
    for j in range(0,rows):
        for i in range(0,cols):

            x = i*width
            y = j*width
            out[x:x+width,y:y+width] = X[counter].reshape(width,width).T
            counter +=1

    img = scipy.misc.toimage(out)

    axes = pyplot.gca()
    figure = pyplot.gcf()

    axes.imshow(img).set_cmap('gray')


def featureNormalize(data):
    mu 	= mean( data, axis=0 )#Normalizitation is done like (x-mu)/sigma
    data_norm 	= data - mu
    sigma   = std( data_norm, axis=0, ddof=1 )
    data_norm 	= data_norm / sigma
    return data_norm, mu, sigma
    
def pca(X):

    covariance = X.T.dot(X)/shape(X)[0]
    U,S,V = linalg.svd(covariance)

    return U,S


def projectData(X_norm,U,K):

    return X_norm.dot(U)[:,:K]

def recoverData(Z,U,K):

    return Z.dot(U[:,:K].T)

def pca100(X):

    X_norm, mu, sigma = featureNormalize( X )

    U, S = pca( X_norm )

    K = 200
    Z = projectData( X_norm, U, K )
    X_rec = recoverData( Z, U, K )

    pyplot.subplot( 1, 2, 1 )
    pyplot.title('Original faces')
    print 'Display original faces\n'
    displayData( X_norm[:100, :] )
    pyplot.subplot( 1, 2, 2 )
    pyplot.title('Recovered faces')
    print 'Display recovered faces\n'
    displayData( X_rec[:100, :] )
    pyplot.show( block=True )

    error = 1 - (sum( S[:K]) / sum( S))#S[:1] beacuse K is 1
    print 'Error should be less than 0.01:\n',error

def runKMeans():

    A = scipy.misc.imread( "bird_small.png" )
    A 			= A / 255.0
    img_size 	= shape( A )

    X = A.reshape( img_size[0] * img_size[1], 3 )
    K = 16
    max_iters = 10


    initial_centroids = kMeansInitCentroids( X, K )
    centroids, idx = runKmeans( X, initial_centroids, max_iters )


    fig = pyplot.figure()
    axis = fig.add_subplot( 111, projection='3d' )

    X_norm, mu, sigma = featureNormalize( X )

    U, S = pca( X_norm )
    Z = projectData( X_norm, U, 2 )
    axis.scatter( Z[:100, 0], Z[:100, 1], c='r', marker='o' )
    pyplot.show(block=True)
    
def main():

    data = scipy.io.loadmat("ex7data1.mat")

    X = data['X']
   
    
    pyplot.plot(X[:,0],X[:,1],'bo')

    pyplot.axis([0.5,6.5,2,8])

    pyplot.axis('equal')

    pyplot.show(block=True)

    X_norm,mu,sigma = featureNormalize(X)

    U,S = pca(X_norm)#U is 2x2

    print 'Top principal components:\n'

    print U

    K=1#1 dimension

    error = 1 - (sum( S[:K]) / sum( S))#S[:1] beacuse K is 1
    print 'Error should be less than 0.01:\n',error

    Z = projectData(X_norm,U,K)

    print 'Value of the first projected example onto the first dimension:\n'

    print Z[0]

    X_rec = recoverData(Z,U,K)

    print 'Data recovered to the original size:\n'

    print X_rec[0]

    mu = mu.reshape(1,2)[0]

    mu_1 = mu + 1.5 * S[0]*U[:,0]

    mu_2 = mu + 1.5 * S[1]*U[:,1]


    pyplot.plot(X[:,0],X[:,1],'bo')

    pyplot.gca().add_line(lines.Line2D(xdata=[mu[0],mu_1[0]],ydata=[mu[1],mu_1[1]],c='r',lw=2))
    pyplot.gca().add_line(lines.Line2D(xdata=[mu[0],mu_2[0]],ydata=[mu[1],mu_2[1]],c='r',lw=2))

    pyplot.axis( [0.5, 6.5, 2, 8] )
    pyplot.axis( 'equal' )

    pyplot.show( block=True )

    for i in range( 0, shape( X_rec)[0] ):
        pyplot.gca().add_line( lines.Line2D( xdata=[X_norm[i,0], X_rec[i,0]], ydata=[X_norm[i,1], X_rec[i,1]], c='g', lw=1, ls='--' ) )	

    pyplot.plot( X_norm[:, 0], X_norm[:, 1], 'bo' )
    #pyplot.title('Original faces')
    pyplot.plot( X_rec[:, 0], X_rec[:, 1], 'ro' )
    #pyplot.title('Recovered faces')
	
    pyplot.axis( 'equal' )
    pyplot.axis( [-4, 3, -4, 3] )

    pyplot.show( block=True )

    print 'Dimensional reduction for K=100\n'

    data = scipy.io.loadmat( "ex7faces.mat" )
    X = data['X']

    pca100(X)

    print 'Run K-means...\n'

    runKMeans()
    

    

    

    

    
    
    
    
    

   

if __name__ == '__main__':
	main()   
