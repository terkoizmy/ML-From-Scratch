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

class LinearRegression() :
      
    def __init__( self, learning_rate = 0.1 , iterations = 10000) :
        """ Set a Learning rate and Iterations
 
        LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
        to minimize the residual sum of squares between the observed targets in
        the dataset, and the targets predicted by the linear approximation.
 
        Parameters
        ------------
            learning_rate : float, default is 0.1
                The amount that the weights are updated during training is
                referred to as the step size
            iterations : int, default is 10000
                An iteration is a term used in machine learning and indicates
                the number of times the algorithm's parameters are updated
        """
        
        self.learning_rate = learning_rate
        self.iterations = iterations
          
    def fit( self, X, Y ) :

        """ Fit linear model.
 
        Parameters
        ------------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values. Will be cast to X's dtype if necessary
        
        Return
        ------------
            self : returns an instance of self. 
        """
        
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros( self.n )
        self.b = 0
        self.X = X
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :
            self.update_weights()
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
        Y_pred = self.predict( self.X )
        
        # calculate gradients  
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
        
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
        
        """ Predict using the linear model.
 
        Parameters
        ------------
            X : array_like or sparse matrix, shape (n_samples, n_features)
                Samples.
        
        Return
        -----------
            C : array, shape (n_samples,)
                Returns predicted values.
        """
        self.pred = X.dot( self.W ) + self.b
        return self.pred
        