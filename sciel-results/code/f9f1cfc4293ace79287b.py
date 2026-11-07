import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of the first 10 
    Legendre polynomials. This basis is orthonormal over [-1, 1], which 
    significantly improves the condition number of the optimization problem 
    compared to raw monomials, allowing BFGS to converge faster and more accurately.
    """
    # Map input x from [-3, 3] to t in [-1, 1] for standard Legendre polynomials
    t = x / 3.0
    
    # Compute the first 10 Legendre polynomials P_0(t) to P_9(t)
    # P_0(t) = 1
    # P_1(t) = t
    # P_{n+1}(t) = ((2n+1)*t*P_n(t) - n*P_{n-1}(t)) / (n+1)
    
    n_terms = 10
    basis = np.zeros((len(t), n_terms))
    
    # P_0
    basis[:, 0] = 1.0
    
    # P_1
    if n_terms > 1:
        basis[:, 1] = t
        
    # P_n for n >= 2
    for n in range(1, n_terms - 1):
        basis[:, n + 1] = ((2 * n + 1) * t * basis[:, n] - n * basis[:, n - 1]) / (n + 1)
        
    # The model is y = sum(params[i] * basis[:, i])
    # This is a matrix-vector product
    y_pred = basis @ params
    
    return y_pred
