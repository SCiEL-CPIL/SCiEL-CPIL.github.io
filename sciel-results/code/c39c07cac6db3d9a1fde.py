import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Radial Basis Function (RBF) network.
    
    The model is a linear combination of 10 Gaussian RBFs:
    f(x, y, z) = sum_{i=0}^{9} p_i * exp(-gamma * ||[x,y,z] - c_i||^2)
    
    The centers c_i are fixed to cover the input domain [0.5, 2]^3.
    The width parameter gamma is fixed to 3.0.
    The parameters p are the weights for each RBF.
    
    This approach allows the model to fit complex, non-polynomial, and non-periodic
    functions that a simple polynomial or trigonometric basis might miss.
    """
    p = params
    
    # Define 10 fixed centers in the input domain [0.5, 2.0]
    # We use the 8 corners of the cube, the center, and one additional point for better coverage
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 1.25]
    ])
    
    # Gamma controls the width of the RBFs
    # A larger gamma means narrower, more local functions
    # A smaller gamma means wider, more global functions
    # 3.0 is a reasonable default for a domain size of 1.5
    gamma = 3.0
    
    # Reshape input to (-1, 3) for vectorized distance calculation
    # x, y, z are 1D arrays of the same length
    inputs = np.column_stack((x, y, z))
    
    # Calculate squared distances from each input point to each center
    # inputs: (N, 3)
    # centers: (10, 3)
    # dist_sq: (N, 10)
    dist_sq = np.sum((inputs[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
    
    # Calculate RBF values
    # rbf_values: (N, 10)
    rbf_values = np.exp(-gamma * dist_sq)
    
    # Compute the linear combination
    # prediction: (N,)
    prediction = rbf_values.dot(p)
    
    return prediction
