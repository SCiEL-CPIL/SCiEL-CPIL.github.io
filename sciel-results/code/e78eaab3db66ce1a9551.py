import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 basis functions.
    The model is linear in parameters, ensuring the optimization problem is convex
    and BFGS will reliably find the global minimum for the given basis.
    
    Basis functions chosen to cover polynomial, trigonometric, exponential, and
    logarithmic behaviors on the domain [0.1, 6].
    """
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    # Define the 10 basis functions
    # 1. Constant
    phi_1 = np.ones_like(x)
    # 2. Linear
    phi_2 = x
    # 3. Quadratic
    phi_3 = x**2
    # 4. Cubic
    phi_4 = x**3
    # 5. Quartic
    phi_5 = x**4
    # 6. Sine
    phi_6 = np.sin(x)
    # 7. Cosine
    phi_7 = np.cos(x)
    # 8. Sine (2x)
    phi_8 = np.sin(2 * x)
    # 9. Exponential decay
    phi_9 = np.exp(-x)
    # 10. Logarithm
    phi_10 = np.log(x)
    
    # Stack into a matrix of shape (N, 10)
    # Each column is a basis function evaluated at all points
    basis = np.column_stack((
        phi_1, phi_2, phi_3, phi_4, phi_5,
        phi_6, phi_7, phi_8, phi_9, phi_10
    ))
    
    # The prediction is the dot product of the basis matrix and the parameters
    # basis: (N, 10), params: (10,) -> result: (N,)
    return basis @ params
