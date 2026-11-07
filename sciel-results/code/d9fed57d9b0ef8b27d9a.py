import numpy as np
import time
from scipy.special import erf, expit # expit not needed, just erf

# Constants for EI calculation
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_2 = 1 / np.sqrt(2)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Expected Improvement (EI).
    
    This implementation includes an explicit deadline guard to ensure the function
    returns valid scores within a strict time limit, falling back to a simpler
    heuristic if the computation exceeds the safety margin.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Start deadline guard timer
    t_start = time.time()
    # Safety margin: assume the evaluator allows ~1s per step, but we want to be < 50ms
    # This is an explicit guard as requested.
    deadline = t_start + 0.050
    
    # Ensure sigma is strictly positive
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Check deadline before heavy computation (though z is cheap)
    if time.time() > deadline:
        # Fallback: return uncertainty (sigma) as a proxy for exploration
        # This ensures we return finite scores quickly.
        return np.nan_to_num(sigma_safe, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute EI manually using erf for precision and speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    # Clip z to avoid numerical issues in exp for very large negative z (underflow to 0 is fine)
    # and very large positive z (erf goes to 1).
    # erf is stable for large inputs.
    z_erf_arg = z * INV_SQRT_2
    phi_z = 0.5 * (1.0 + erf(z_erf_arg))
    
    # Calculate exp term. For large |z|, exp(-0.5*z^2) underflows to 0, which is correct.
    # We can clip z for the exp calculation to prevent warnings, though numpy handles it.
    z_sq_half = 0.5 * z * z
    # Clip to prevent overflow in exp if z was imaginary (not possible here) or just for safety
    # exp(-1000) is 0.
    pdf_z = np.exp(-z_sq_half) / SQRT_2PI
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Check deadline again
    if time.time() > deadline:
        # If we are running slow, return what we have so far (which is the full calc)
        # But the directive says "return the best result found". Since we computed it, return it.
        pass
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
