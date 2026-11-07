import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Use a 9th-degree Chebyshev polynomial basis on the scaled input.
    Chebyshev polynomials T_k(t) are orthogonal on [-1, 1] with respect to the
    weight 1/sqrt(1-t^2), and they provide excellent approximation properties
    for smooth functions. By scaling x from [-2,2] to [-1,1], we get a well-
    conditioned, linear-in-parameters model that BFGS can fit to the global
    minimum of the least-squares problem efficiently.
    
    The 10 basis functions are T_0(t), T_1(t), ..., T_9(t) where t = x/2.
    This spans the space of all polynomials up to degree 9, which can capture
    a wide variety of smooth target functions including polynomials, and
    approximates non-polynomial smooth functions well.
    
    Since the model is linear in the 10 parameters, BFGS converges to the
    global minimum (the least-squares solution) without local minima issues.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if params.ndim != 1:
        params = params.flatten()
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x from [-2, 2] to [-1, 1] for Chebyshev domain
        t = x / 2.0
        
        # Compute Chebyshev polynomials T_0 through T_9 using the recurrence:
        # T_0(t) = 1
        # T_1(t) = t
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        
        T0 = np.ones_like(t)
        T1 = t.copy()
        
        basis = [T0, T1]
        
        for k in range(2, 10):
            Tk = 2.0 * t * basis[k-1] - basis[k-2]
            basis.append(Tk)
        
        # Stack into basis matrix (n x 10)
        basis_matrix = np.column_stack(basis)
        
        # Linear model: y = basis_matrix @ params
        result = basis_matrix @ params
        
        # Ensure finite output
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x)
            
        return result
        
    except Exception:
        return np.zeros_like(x)
