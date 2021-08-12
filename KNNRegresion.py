import numpy as np

def Score(actual, predict):
    """ Calculation of result with MAE and MSE Score.

    Parameter
    ----------
        actual : array of actual data from test data
        predict : array of predict data from regression
    """
    d = actual - predict
    mse_f = np.mean(d**2)
    mae_f = np.mean(abs(d))

    print("Results by manual calculation:")
    print("MAE:",mae_f)
    print("MSE:", mse_f)

class KNNRegression() : 
      
    def __init__( self, K = 5 ) :
        """ Regression based on k-nearest neighbors.

        The target is predicted by local interpolation of the targets
        associated of the nearest neighbors in the training set.

        Parameters
        ------------
        K : int, optional (default = 5)
            Number of neighbors to use by default for kneighbors queries.
        """
        self.K = K
          
    def fit( self, X_train, Y_train ) :
        """ Fit the model using X as training data and y as target values.

        Parameters
        ------------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
        or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values, array of float values, shape = [n_samples]
        or [n_samples, n_outputs]
        """

        self.X_train = X_train
        self.Y_train = Y_train
        self.m, self.n = X_train.shape
        return self

    def predict( self, X_test ) :
        """Predict the target for the provided data

        Parameters
        ------------
        X : array-like, shape (n_queries, n_features), or (n_queries, n_indexed).

        Returns
        ------------
        y : array of int, shape = [n_queries] or [n_queries, n_outputs]
            Target values
        """

        self.X_test = X_test
        self.m_test, self.n = X_test.shape
        Y_predict = np.zeros( self.m_test )
          
        for i in range( self.m_test ) :
            x = self.X_test[i]
            neighbors = np.zeros( self.K )
            neighbors = self.find_neighbors( x )
            Y_predict[i] = np.mean( neighbors )
              
        return Y_predict
            
    def find_neighbors( self, x ) :
        """calculate all the euclidean distances between current test
        
        Parameters
        ------------
        X : array-like, shape (n_queries, n_features), or (n_queries, n_indexed).

        Return
        ------------
        Y_train_sorted with K Neighbors
        """
        euclidean_distances = np.zeros( self.m )
          
        for i in range( self.m ) :
            d = self.euclidean( x, self.X_train[i] )
            euclidean_distances[i] = d
          
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]
              
    def euclidean( self, x, x_train ) :
        """Formula for calculate a distance"""
        return np.sqrt( np.sum( np.square( x - x_train ) ) )
    