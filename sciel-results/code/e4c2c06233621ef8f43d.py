import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Linear model with a fixed softplus basis.
    The basis consists of 10 softplus functions with fixed centers and slopes
    arranged symmetrically over the domain [-2, 2]. Only the weights are learned,
    making the problem linear in params. BFGS will converge to the global minimum
    of this linear least-squares problem.
    
    Parameter allocation (10 total):
    - p0-p9: weights for the 10 softplus basis functions
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Fixed centers and slopes for the softplus basis
        # Centers symmetrically distributed in [-2, 2]
        centers = np.linspace(-2.0, 2.0, 10)
        
        # Fixed slopes: use a moderate slope for good approximation properties
        # Slopes increase towards the edges for better resolution where the function
        # might have sharper features, but keep them fixed for linearity
        slopes = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 6.0, 5.0, 4.0, 3.0])
        
        # Compute softplus basis: softplus(z) = log(1 + exp(z))
        # For numerical stability, use: softplus(z) = max(z, 0) + log1p(exp(-|z|))
        basis = np.zeros((len(x), 10))
        for i in range(10):
            z = slopes[i] * (x - centers[i])
            # Numerically stable softplus
            basis[:, i] = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
        
        # Linear combination of basis functions with learned weights
        result = basis @ params
        
        return result
        
    except Exception:
        return np.zeros_like(x)
