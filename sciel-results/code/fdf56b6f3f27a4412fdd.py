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
    
    # Strategy: Parameterized Gaussian Mixture Model (GMM)
    # We use 5 Gaussian "bumps".
    # Parameters:
    # p[0..4]: Centers (mu)
    # p[5..9]: Widths (sigma)
    # Amplitudes are fixed to 1.0 to keep the model linear in the 
    # sense that we are summing bumps with fixed height, but wait, 
    # if amplitudes are fixed, BFGS can only move them.
    # Actually, to allow BFGS to fit better, we should probably 
    # allow amplitudes to vary? No, we only have 10 params.
    # If we use 5 bumps, we need 3 params per bump (mu, sigma, amp). 
    # 5*3 = 15 params. We only have 10.
    # So we must share parameters or fix some.
    
    # Option A: 5 bumps, fixed amplitude 1.0. 
    # BFGS moves centers and widths. This is non-linear in params.
    # This might be harder for BFGS than linear regression, but 
    # more flexible locally.
    
    # Option B: 2 bumps, 5 params each? No, 2*5=10.
    # Let's stick to 5 bumps with fixed amplitude 1.0.
    # The sum of 5 unit-height Gaussians can approximate a wide range 
    # of shapes if centers and widths are optimized.
    
    # However, if the target function has large amplitude, 
    # fixed amplitude 1.0 might be too small. 
    # We can scale the output? No, that would be a parameter.
    
    # Let's try a hybrid: 
    # Use 5 Gaussians. 
    # p[0..4]: Centers
    # p[5..9]: Log-widths (to ensure positive width) -> exp(p)
    # Amplitudes are fixed at 1.0.
    
    # If the target function is, say, 10*x, this model might struggle 
    # if it can't scale amplitude. 
    # But BFGS starts at all ones. 
    # If we start at ones, centers are at 1, widths are exp(1)=e.
    # The sum of 5 such Gaussians is a broad, low peak.
    
    # To be safe, let's use a mix:
    # 2 Gaussians (6 params: mu1, sig1, amp1, mu2, sig2, amp2)
    # + Linear term (2 params: slope, intercept)
    # + Quadratic term (1 param)
    # + Cubic term (1 param)
    # Total: 6 + 2 + 1 + 1 = 10 params.
    
    # This combines local flexibility (Gaussians) with global trends (Polynomials).
    
    # p[0]: mu1
    # p[1]: sig1 (raw, will be abs+eps)
    # p[2]: amp1
    # p[3]: mu2
    # p[4]: sig2 (raw, will be abs+eps)
    # p[5]: amp2
    # p[6]: linear coeff
    # p[7]: intercept
    # p[8]: quadratic coeff
    # p[9]: cubic coeff
    
    mu1 = p[0]
    sig1 = np.abs(p[1]) + 1e-6
    amp1 = p[2]
    
    mu2 = p[3]
    sig2 = np.abs(p[4]) + 1e-6
    amp2 = p[5]
    
    lin = p[6]
    const = p[7]
    quad = p[8]
    cubic = p[9]
    
    # Compute Gaussians
    # Use exp(-0.5 * ((x - mu)/sig)^2)
    # To avoid overflow in exp if sig is very small and x-mu is large?
    # (x-mu)^2 / sig^2 can be large. exp(-large) -> 0. Safe.
    # If sig is 1e-6, (x-mu)^2 / 1e-12 is huge. exp(-huge) is 0.
    # Only risk is if sig is 0, which we prevented.
    
    g1 = amp1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2)
    
    # Polynomial part
    poly = const + lin * x + quad * x**2 + cubic * x**3
    
    result = g1 + g2 + poly
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
