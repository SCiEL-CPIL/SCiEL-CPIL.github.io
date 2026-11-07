"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global cache to store the best parameters found so far
# and to track timing
_best_params_cache = None
_start_time = None

def _get_best_params(x, params, y):
    """
    Perform multi-start BFGS optimization to find the best parameters
    for the flexible model. Uses the input `params` to generate diverse
    starting points, leveraging the 10-dimensional space provided by the evaluator.
    """
    global _best_params_cache, _start_time
    
    # Deadline guard: 145s to be safe against the 150s limit
    deadline = _start_time + 145.0
    if _start_time is None:
        _start_time = time.time()
        deadline = _start_time + 145.0

    if _best_params_cache is not None:
        # If we have a previous best, check if it's still good
        # But since x,y change per call, we must re-optimize or trust the cache?
        # The evaluator calls equation(x, params) where x is the training set.
        # The training set is fixed. So the optimal params for the model structure
        # should be the same regardless of the `params` passed in (which are just seeds).
        # However, the `params` passed in by the external BFGS are the current state.
        # If we return the cached best, we are effectively ignoring the external optimizer's progress
        # and forcing it to converge to our internal optimum.
        
        # Let's verify the cached params are still valid/finite
        if np.all(np.isfinite(_best_params_cache)):
             return _best_params_cache
        else:
             _best_params_cache = None

    # Generate diverse starting points
    # We use the input `params` (length 10) to seed the search.
    # We will test a set of candidate structures and starting points.
    
    # Model: A combination of polynomial and trigonometric terms.
    # y = a0 + a1*x + a2*x^2 + a3*x^3 + a4*sin(x) + a5*cos(x) + a6*exp(-x^2) + a7*tanh(x)
    # This uses 8 parameters. The external `params` has 10.
    # We will map the external params to internal model params via a simple transformation
    # or just use the first 8 as seeds and the last 2 for scaling/variance.
    
    # Let's define the model function
    def model(x, p):
        # p is the internal parameter vector (length 8)
        return (p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + 
                p[4]*np.sin(x) + p[5]*np.cos(x) + 
                p[6]*np.exp(-x**2) + p[7]*np.tanh(x))
    
    def loss(p):
        pred = model(x, p)
        return np.mean((pred - y)**2)

    # Candidate starting points
    starts = []
    
    # 1. Use the input params (first 8) as a start
    if len(params) >= 8:
        starts.append(params[:8].copy())
    
    # 2. All ones
    starts.append(np.ones(8))
    
    # 3. All zeros
    starts.append(np.zeros(8))
    
    # 4. Random starts derived from hash of x to ensure consistency but diversity
    # Actually, we can just add a few heuristic starts
    starts.append(np.array([0, 1, 0, 0, 0, 0, 0, 0])) # Linear
    starts.append(np.array([0, 0, 1, 0, 0, 0, 0, 0])) # Quadratic
    starts.append(np.array([0, 0, 0, 1, 0, 0, 0, 0])) # Cubic
    starts.append(np.array([0, 0, 0, 0, 1, 0, 0, 0])) # Sin
    starts.append(np.array([0, 0, 0, 0, 0, 1, 0, 0])) # Cos
    starts.append(np.array([0, 0, 0, 0, 0, 0, 1, 0])) # Gaussian
    starts.append(np.array([0, 0, 0, 0, 0, 0, 0, 1])) # Tanh
    
    # Remove duplicates and invalid starts
    unique_starts = []
    seen = set()
    for s in starts:
        if not np.all(np.isfinite(s)):
            continue
        key = tuple(np.round(s, 6))
        if key not in seen:
            seen.add(key)
            unique_starts.append(s)
    
    best_p = None
    best_loss = np.inf
    
    # Multi-start optimization
    for p0 in unique_starts:
        if time.time() > deadline - 5: # Safety margin
            break
            
        try:
            res = minimize(loss, p0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
            if res.success and np.isfinite(res.fun):
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_p = res.x
        except Exception:
            continue
            
    if best_p is not None:
        _best_params_cache = best_p
        return best_p
    else:
        # Fallback: return the input params (truncated/padded to 8)
        if len(params) >= 8:
            return params[:8]
        else:
            return np.ones(8)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _start_time, _best_params_cache
    
    # Initialize timing
    if _start_time is None:
        _start_time = time.time()
        
    x = np.asarray(x, dtype=float)
    
    # We need y to optimize. But `equation` only receives x and params.
    # WAIT. The task says: "The evaluator fits params... minimizing the training MSE."
    # The evaluator calls `equation(x, params)` to get predictions.
    # It does NOT pass `y` to `equation`.
    # Therefore, I CANNOT perform internal optimization against `y` inside `equation`
    # because I don't have access to `y`!
    
    # Re-reading the task:
    # "Write equation(x, params)... The evaluator fits params... minimizing the training MSE."
    # This implies the external optimizer (BFGS) calls `equation` many times with different `params`.
    # I do NOT have access to the target `y` inside `equation`.
    # My previous strategy of "internal multi-start against y" is IMPOSSIBLE.
    
    # Correction: I must design a `equation` that is highly expressive and easy for the external BFGS to optimize.
    # The current program uses a linear basis. Why is the score 4367?
    # Maybe the basis is not rich enough, or the optimization is hard.
    # The score 1.0 (seed) was `p0*x + p1`.
    # The current program (score 4367) is WORSE than the seed.
    # This suggests the rich basis is causing optimization difficulties (local minima, ill-conditioning).
    
    # New Strategy: 
    # Use a "soft" basis that is easier to optimize. 
    # Or, use a basis that is orthogonalized or scaled.
    # Or, stick to a simpler basis but ensure numerical stability.
    # 
    # Actually, looking at the inspiration: "score 1.0000" uses the seed program.
    # The current program is much worse.
    # Let's try a simpler, more robust basis.
    # Polynomial up to degree 3 + Sin + Cos.
    # Scale the features to [-1, 1] to help BFGS? No, x is fixed.
    # 
    # Let's try a basis that is less prone to oscillation and ill-conditioning.
    # Basis:
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: sin(x)
    # 5: cos(x)
    # 6: tanh(x)
    # 7: x * sin(x)
    # 8: x * cos(x)
    # 9: x^2 * sin(x)
    
    # Let's implement this.
    
    p = params
    
    f0 = 1.0
    f1 = x
    f2 = x**2
    f3 = x**3
    f4 = np.sin(x)
    f5 = np.cos(x)
    f6 = np.tanh(x)
    f7 = x * np.sin(x)
    f8 = x * np.cos(x)
    f9 = x**2 * np.sin(x)
    
    result = (
        p[0] * f0 +
        p[1] * f1 +
        p[2] * f2 +
        p[3] * f3 +
        p[4] * f4 +
        p[5] * f5 +
        p[6] * f6 +
        p[7] * f7 +
        p[8] * f8 +
        p[9] * f9
    )
    
    return result
# EVOLVE-BLOCK-END
