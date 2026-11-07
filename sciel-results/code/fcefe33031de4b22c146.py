# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: 10 RBFs with a single shared learnable width.
    # This is a structural change from the seed (5 RBFs + 4th order poly).
    # By using 10 centers, we get finer local resolution.
    # By sharing the width, we reduce the parameter space complexity for BFGS,
    # preventing the "spiky" local minima that can occur with independent widths.
    # The width is log-parameterized (exp) to ensure positivity and smooth gradient behavior.
    
    # Centers: 10 evenly spaced points in [-3, 3]
    # np.linspace(-3, 3, 10)
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Parameters:
    # p[0..9]: Weights for the 10 RBFs
    # The width is NOT in the 10 params? Wait, the task says "params, an array of length 10".
    # The evaluator fits params of length 10.
    # So I must fit the width into one of the 10 params, or fix it.
    # The directive says: "using a single learnable width parameter shared by all kernels".
    # Since I only have 10 params for 10 centers, I have 0 params left for width if I use all 10 for weights.
    # Option A: Use 9 weights + 1 width param. (9 centers)
    # Option B: Use 10 weights + fixed width. (Not "learnable width")
    # Option C: Use 10 params where one is width, 9 are weights.
    
    # Let's look at the seed: 5 RBF weights + 5 poly coeffs. Total 10.
    # If I want 10 RBFs, I need 10 weights. That leaves 0 for width.
    # The directive says "Replace the fixed analytic basis with a linear combination of 10 RBFs at fixed, evenly spaced centers, using a single learnable width parameter".
    # This implies I should perhaps use 9 weights and 1 width param? Or is the "10 RBFs" a suggestion that might require dropping one weight?
    # "10 RBFs" usually means 10 basis functions. If I have 10 basis functions, I need 10 coefficients.
    # If I must have a learnable width, I must sacrifice one coefficient.
    # Let's use 9 RBF centers and 1 width parameter.
    # Or, more likely, the directive implies I should structure the 10 params to include the width.
    # Let's use 9 centers to leave room for the width param.
    # Actually, let's re-read carefully: "linear combination of 10 RBFs... using a single learnable width parameter".
    # This is slightly contradictory if I only have 10 slots.
    # I will interpret this as: Use the 10 params to define the model. 
    # To satisfy "learnable width", I will use 9 weights and 1 width param.
    # 9 centers is still a significant improvement over 5.
    
    # Let's try 9 centers.
    # Centers: linspace(-3, 3, 9)
    centers = np.linspace(-3.0, 3.0, 9)
    
    # p[0..8]: Weights
    # p[9]: Log-width (sigma_log). sigma = exp(p[9])
    
    weights = p[0:9]
    sigma_log = p[9]
    
    # Clamp sigma to avoid numerical issues (exp overflow/underflow)
    # If sigma_log is very large, sigma is huge (flat). If very small, sigma is tiny (spiky).
    # BFGS starting from ones: p[9]=1 => sigma = e^1 ~ 2.7. This is a reasonable starting width for domain size 6.
    
    # To be safe against exp overflow in BFGS if it wanders, we can clip the log-sigma.
    # However, BFGS should handle it. Let's add a small clip just in case for stability.
    sigma_log_clipped = np.clip(sigma_log, -5.0, 5.0)
    sigma = np.exp(sigma_log_clipped)
    
    # Compute RBF features
    # x is shape (N,), centers is shape (9,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF: exp(-0.5 * (diff / sigma)^2)
    # Ensure sigma is not zero
    if sigma < 1e-6:
        sigma = 1e-6
        
    rbf = np.exp(-0.5 * (diff / sigma)**2)
    
    # Linear combination
    # rbf @ weights gives shape (N,)
    result = rbf @ weights
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        # Use a simple linear fallback using the first two params
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
