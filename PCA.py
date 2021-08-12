import numpy as np
class PCA() :

    def __init__( self, n_components):
        """ Principal component analysis (PCA).

        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space.
 
        Parameters
        ------------
            n_components : int, float, None or str
            Number of components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """ Fit the model with X.

        Parameters
        ------------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
        and n_features is the number of features.

        Returns
        ------------
        self : object
            Returns the instance itself. 
        """

        # Mean centering
        self.mean = np.mean(X.T, axis=1)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ------------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
        and n_features is the number of features.

        Returns
        ------------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        ------------
        >>> import numpy as np
        >>> import PCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca.fit(X)
        >>> ipca.transform(X) # doctest: +SKIP
        
        """
        X = X - self.mean
        return np.dot(X, self.components.T)