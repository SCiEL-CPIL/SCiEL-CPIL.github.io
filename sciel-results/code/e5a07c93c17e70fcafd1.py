import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a hidden function using a linear combination of Chebyshev Polynomials.
    
    The 10 parameters are fitted by the evaluator using BFGS.
    The basis consists of Chebyshev polynomials of the first kind, mapped to the domain [-3, 3].
    Chebyshev polynomials provide a well-conditioned global basis for approximating smooth functions,
    often outperforming raw polynomials and localized RBFs for general symbolic regression tasks 
    due to their orthogonality and minimal max-norm error properties.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for standard Chebyshev definition
    t = x / 3.0
    
    # Initialize result
    result = np.zeros_like(x)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T_k_minus_2 = np.ones_like(x)  # T_0
    T_k_minus_1 = t                 # T_1
    
    # Add T_0 contribution
    result += params[0] * T_k_minus_2
    
    # Add T_1 contribution
    if len(params) > 1:
        result += params[1] * T_k_minus_1
        
    # Compute and add T_2 to T_9
    for k in range(2, 10):
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        result += params[k] * T_k
        
        # Update for next iteration
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k

    return result
