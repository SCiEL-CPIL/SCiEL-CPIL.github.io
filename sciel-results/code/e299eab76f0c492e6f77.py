# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using a small neural network with 20 hidden units,
    ReLU activation, and a single output unit.
    
    The 10 available parameters are used to parameterize the network in a way
    that ensures differentiability for BFGS optimization.
    
    Strategy:
    - We use a fixed deterministic initialization for most weights to ensure
      the function remains closed-form and differentiable.
    - The 10 parameters control key aspects of the network:
      * params[0:5] - First set of 5 hidden layer weights (shared across units)
      * params[5:10] - Second set of 5 hidden layer biases (shared across units)
    
    Actually, let me reconsider. With only 10 parameters, we can't have a full 
    20-hidden-unit network with all weights free. Let's use a more clever approach:
    
    We'll use a fixed-basis approach where the basis functions are generated 
    deterministically, and the 10 parameters are the linear combination coefficients.
    This makes the problem convex in the parameters, which BFGS can handle well.
    
    Basis functions: 10 diverse functions that can approximate many target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 basis functions that span a rich function space
    # These are chosen to be diverse and cover different frequency scales
    # and functional forms
    
    # Basis 0: constant
    b0 = np.ones_like(x)
    
    # Basis 1: linear
    b1 = x
    
    # Basis 2: quadratic
    b2 = x**2
    
    # Basis 3: cubic
    b3 = x**3
    
    # Basis 4: quartic
    b4 = x**4
    
    # Basis 5: sin(x)
    b5 = np.sin(x)
    
    # Basis 6: cos(x)
    b6 = np.cos(x)
    
    # Basis 7: sin(2x)
    b7 = np.sin(2 * x)
    
    # Basis 8: cos(2x)
    b8 = np.cos(2 * x)
    
    # Basis 9: exp(-x^2) - Gaussian bump
    b9 = np.exp(-x**2)
    
    # Stack basis functions: shape (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with the 10 parameters
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
