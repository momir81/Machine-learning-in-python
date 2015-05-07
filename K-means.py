import sys

from numpy import *
import scipy.misc,scipy.io,scipy.optimize,scipy.cluster.vq

from matplotlib import pyplot,cm,colors

from mpl_toolkits.mplot3d import Axes3D



def computeCentroids(X,idx,K):

    m,n = shape(X)
    #K is 3

    centroids = zeros((K,n))
    

    data = c_[X, idx]# futher in the code assigns each cluster to specific points, add idx column to X to the right side 300x3
    for k in range(1,K+1):
        
        #temp = (data[:,n] == k).nonzero()
        #print 'temp',temp
        
        temp2 = (idx==k).nonzero()

        #print 'temp2 shape',shape(temp2[0])

        temp1 = data[temp2[0]]
        
        
 
        count = shape(temp1)[0]
        for j in range(0,n):
            centroids[k-1,j] = sum(temp1[:,j])/count#3x2
        
        
    return centroids
        #temp = data[data[:,n] == k]#extract n column from data and compare with 1,2,3 clsuters 
   
def findClosestCentroids(X,init_centroids):

    K = shape(init_centroids)[0]#3

    m = shape(X)[0]#300

    idx = zeros((m,1))#300x1

    for i in range(0,m):
        min_dist = 999
        min_index = 0
        #idx[i] is the index of the centroid that is closest to x[i], and init_centroid[j] is the position of the jth centroid 
        # if x[1],x[2] are assigned to cluster 2 this means idx[1],idx[2] are 2
        for j in range(0,K):
            
            dist = X[i] - init_centroids[j]#2 vector i ide od 1 do 300 a j je 1, pa ponovo oduzme svih 300 od jednom,
            #pa svih trista od frugog i 
            #print 'cost1\n',shape(cost1)

            dist = dist.dot(dist).T#this is an array of number size3x1 [[4],[6],[7]], where these numers are square products of cost1

            if dist< min_dist:
                min_index = j
                min_dist = dist
        idx[i]=min_index
    #print 'idx',idx[0:50]
            #min_cost=argmin(idx,axis=1)+1#racuna minnum i njihove idekse smesta u idx
            

    return idx+1
    
def runKmeans(X,init_centroids,max_iters,plot=False):

    K = shape(init_centroids)[0]
    centroids 	= copy( init_centroids )

    

    idx = None

    print 'shape centroid',shape(centroids)

    for iteration in range(0,max_iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,K)

        if plot is True:
            data = c_[X, idx]

            cluster_1 = data[data[:,2] == 1]
            pyplot.plot(cluster_1[:,0],cluster_1[:,1],'ro',color='red',markersize=5)

            cluster_2 = data[data[:,2] == 2]
            pyplot.plot(cluster_2[:,0],cluster_2[:,1],'go',color='green',markersize=5)

            cluster_3 = data[data[:,2] == 3]
            pyplot.plot(cluster_3[:,0],cluster_3[:,1],'bo',color='blue',markersize=5)

            pyplot.plot(centroids[:,0],centroids[:,1],'k*',markersize=17)

    pyplot.show(block=True)

    return centroids,idx
            

def kMeansInitCentroids(X,K):

    return random.permutation(X)[:K]



def main():
    data = scipy.io.loadmat("ex7data2.mat")

    X = data['X']#300x2

    K = 3

    init_centroids = array([[3,3],[6,2],[8,5]])#3x2

    idx = findClosestCentroids(X,init_centroids)
    
    print 'Closest centroids for the first 3 examples:\n'

    print idx[0:3]

    print '\nThe closest centroids should be 1,3,2\n'

    print 'Computing centroids means\n'

    centroids = computeCentroids(X,idx,K)

    print 'Centroids computed: n'

    print centroids

    print 'Run K-means...'

    max_iters = 10

    runKmeans(X,init_centroids,max_iters,plot=True)

    print 'Centroids random initialization\n'

    initCentroids = kMeansInitCentroids(X,K)

    print 'init centroids\n'

    print initCentroids

    img = scipy.misc.imread("bird_small.png")

    axes = pyplot.gca()
    figure = pyplot.gcf()
    axes.imshow(img)

    pyplot.show(block=True)

    print 'before scalling',shape(img)#128x128x3

    

    img = img/255.0#scalling image pixels between 0 and 1

    

    print 'after scalling',shape(img)#128x128x3

    img_size = shape(img)

    X = img.reshape(img_size[0]*img_size[1],3)#pixels and color code

    print 'shape X after reshape',shape(X)#16384,3

    K =20
    max_iters = 10

    initial_centroids = kMeansInitCentroids( X, K )
    centroids,idx = runKmeans(X,initial_centroids,max_iters)

    m = shape(X)[0]

    X_recovered = zeros(shape(X))

    for i in range(0,m):
        k = int(idx[i]) - 1
        X_recovered[i] = centroids[k]

    X_recovered = X_recovered.reshape(img_size[0],img_size[1],3)

    print 'X_recovered shape',shape(X_recovered)#128x128x3

    axes = pyplot.gca()
    figure = pyplot.gcf()
    axes.imshow(X_recovered)

    pyplot.show(block=True)

    

    
    
    

    
    






if __name__ == '__main__':
	main()
