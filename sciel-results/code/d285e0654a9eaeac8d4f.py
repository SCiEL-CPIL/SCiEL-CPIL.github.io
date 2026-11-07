import numpy as np
import time

# Global state for the self-contained optimizer
_state = {
    'best_params': None,
    'best_loss': np.inf,
    'start_time': None,
    'initialized': False
}

def _fit_params(x, y):
    """
    Perform a multi-start optimization to find the best 10 parameters.
    This is called when the evaluator first provides y (during training).
    We cache the result so subsequent calls (test) are fast.
    """
    global _state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    # Deadline guard: aim to finish before 145s to be safe
    deadline = _state['start_time'] + 140.0
    
    # Construct the design matrix once (it doesn't change during optimization)
    # Basis: scaled polynomials, scaled trig, scaled exp, tanh
    # Scaling to ~O(1) magnitude on [-3,3] improves conditioning.
    
    x_arr = np.asarray(x, dtype=float)
    N = len(x_arr)
    
    # Define bases carefully scaled
    # x is in [-3, 3]. x^3 is up to 27. x^2 up to 9. 
    # Let's normalize columns to unit norm or similar to help BFGS.
    
    f0 = np.ones_like(x_arr)
    f1 = x_arr / 3.0  # Scale x to [-1, 1]
    f2 = x_arr**2 / 9.0  # Scale x^2 to [0, 1]
    f3 = x_arr**3 / 27.0  # Scale x^3 to [-1, 1]
    f4 = np.sin(x_arr)
    f5 = np.cos(x_arr)
    f6 = np.tanh(x_arr)
    f7 = np.exp(-x_arr**2 / 4.0)  # Wider Gaussian
    f8 = x_arr * np.sin(x_arr) / 3.0
    f9 = x_arr * np.cos(x_arr) / 3.0
    
    # Build matrix
    basis = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # If the matrix is ill-conditioned, regularization helps, but we only have 10 params.
    # Let's stick to direct least squares as a strong baseline for each start, 
    # or just run BFGS on the MSE.
    
    # Since the model is linear in params, we can actually solve it analytically!
    # y = X @ p  =>  p = (X^T X)^-1 X^T y
    # This is much faster and more robust than BFGS for a linear model.
    # However, the task requires the external BFGS to fit params? 
    # "The evaluator fits params... minimizing the training MSE."
    # BUT, the evaluator calls equation(x, params). It does NOT call equation(x, y).
    # Wait, the interface is equation(x, params) -> predictions.
    # The evaluator minimizes over params.
    # So I CANNOT see y in the equation function.
    
    # CORRECTION: 
    # The previous "inspiration" 1 claimed to do internal optimization.
    # How? The interface is equation(x, params). 
    # If I want to do internal optimization, I need y.
    # The standard contract is that equation(x, params) just predicts.
    # The evaluator does:
    # 1. Get x_train, y_train
    # 2. Define loss(params) = MSE(equation(x_train, params), y_train)
    # 3. Minimize loss
    # 4. Get best_params
    # 5. Call equation(x_test, best_params)
    
    # So I CANNOT fit the model inside equation unless I cache state from previous calls.
    # But the first call will be with x_train and some initial params.
    # I don't have y_train on the first call.
    
    # UNLESS... I use the fact that the evaluator calls equation multiple times during the minimize loop.
    # But I still don't have y in any of those calls.
    
    # RE-READING TASK:
    # "Write equation(x, params) ... The evaluator fits params ... minimizing the training MSE."
    # This implies I am a black-box function. I do NOT receive y.
    
    # Therefore, the "internal optimization" strategy from Inspiration 1 is IMPOSSIBLE 
    # unless the interface allows passing y, which it doesn't.
    # Or, unless I use a hack where I assume the "params" vector provided by the optimizer 
    # contains enough information? No, that's circular.
    
    # Wait, looking at Inspiration 1 again:
    # "defines _get_best_params, equation"
    # Maybe the evaluator interface is different?
    # "equation(x, params: np.ndarray) -> np.ndarray"
    # It strictly says this.
    
    # So, I must provide a good basis such that the EXTERNAL BFGS finds the right params.
    # The current program uses a linear basis.
    # The problem with linear basis + BFGS is that if the basis functions are linearly dependent 
    # or ill-conditioned, BFGS might struggle.
    
    # Key Insight:
    # If the model is linear in params, the optimal params can be found in closed form.
    # But I can't compute them because I don't have y.
    # I must rely on BFGS.
    # To help BFGS, I should:
    # 1. Scale features to improve conditioning.
    # 2. Ensure the basis functions are not nearly linearly dependent.
    
    # Let's refine the basis.
    # Current: 1, x, x^2, x^3, sin(x), cos(x), tanh(x), exp(-x^2), x*sin(x), x*cos(x)
    
    # Potential issues:
    # - cos(x) is even, x^2 is even, x^4 (not present) is even.
    # - sin(x) is odd, x is odd, x^3 is odd.
    # - The set is mixed.
    
    # Let's try a more orthogonal set or a richer set.
    # With only 10 params, we are limited.
    
    # What if the target function is high frequency?
    # sin(5x), cos(5x)?
    # My current basis only has frequency 1.
    # If the target is sin(3x), x*sin(x) doesn't capture it well.
    
    # Strategy: Increase the "frequency" coverage of the basis.
    # Use Chebyshev polynomials? They are orthogonal on [-1, 1].
    # Map x from [-3, 3] to [-1, 1]: t = x/3.
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # ...
    # T9(t)
    
    # Chebyshev polynomials are excellent for approximation and well-conditioned.
    # Let's switch to the first 10 Chebyshev polynomials of the first kind.
    
    def chebyshev(n, t):
        # Recursive or closed form
        # cos(n * arccos(t))
        # Using np is fine for small N.
        return np.cos(n * np.arccos(np.clip(t, -1, 1)))
    
    t = x_arr / 3.0
    
    basis = np.empty((N, 10))
    for i in range(10):
        basis[:, i] = chebyshev(i, t)
    
    # The basis is now orthogonal (in a continuous sense) and well-scaled.
    # This should make the Hessian of the MSE much better conditioned for BFGS.
    
    pred = basis @ p
    
    # Check for finite
    if not np.all(np.isfinite(pred)):
        # Return something safe to avoid 1e30 penalty if params explode
        # But BFGS should prevent this.
        pass
        
    return pred

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]
        
    # Map x to [-1, 1]
    t = x / 3.0
    t = np.clip(t, -1.0, 1.0) # arccos domain
    
    # Precompute arccos once
    theta = np.arccos(t)
    
    # Compute Chebyshev polynomials T_k(t) = cos(k * theta)
    # k = 0..9
    k = np.arange(10)
    # Shape: (10, N)
    angles = k[:, np.newaxis] * theta[np.newaxis, :]
    basis = np.cos(angles) # (10, N)
    
    # prediction = sum_i p[i] * T_i(x)
    # pred_j = sum_i p[i] * basis[i, j]
    # pred = p @ basis  -> shape (N,)
    
    pred = p @ basis
    
    return pred
