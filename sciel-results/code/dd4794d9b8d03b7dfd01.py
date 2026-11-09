import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Spectral (Fourier) basis.
    
    Strategy:
    The target is likely the acceleration or a state derivative of a nonlinear oscillator.
    Oscillators exhibit periodic behavior. A polynomial basis (like the previous Duffing model)
    can suffer from ill-conditioning and poor convergence in BFGS when trying to fit 
    oscillatory data, especially if the "oscillation" is not a simple low-degree polynomial.
    
    Instead, we use a truncated 2D Fourier series in the state variables (x, v).
    This provides a global approximation with smooth, bounded basis functions,
    which is often better conditioned for gradient-based optimizers like BFGS.
    
    The basis functions are chosen to capture the fundamental modes and some harmonics.
    We have 10 parameters, so we select 10 basis functions:
    1. 1 (Constant/DC term)
    2. sin(x)
    3. cos(x)
    4. sin(v)
    5. cos(v)
    6. sin(2*x)
    7. cos(2*x)
    8. sin(2*v)
    9. cos(2*v)
    10. sin(x+v)  (Interaction term)
    
    This covers the primary periodic structures in x and v and their coupling.
    """
    
    p = params
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Compute basis functions
    # Note: We assume x and v are within a range where these frequencies are relevant.
    # Given x in [-0.5, 0.5] and v in [-0.2, 0.3], fundamental frequencies (pi or 2pi scaling) 
    # might need adjustment, but standard sin/cos with argument x and v directly 
    # captures the local oscillatory structure if the "period" is roughly 2pi in the normalized space,
    # or we can scale the arguments. 
    # However, without knowing the exact physical scale, using x and v directly is a safe start.
    # If the oscillation is faster, we might need sin(k*x). 
    # Let's stick to the direct arguments first, as the optimizer can scale the coefficients.
    # Actually, for better conditioning, it's often better to let the *frequencies* be fixed 
    # and the *amplitudes* be fitted.
    
    # Basis 0: Constant
    b0 = 1.0
    
    # Basis 1-3: First harmonics of x
    b1 = np.sin(x)
    b2 = np.cos(x)
    
    # Basis 3-4: First harmonics of v
    b3 = np.sin(v)
    b4 = np.cos(v)
    
    # Basis 5-6: Second harmonics of x
    b5 = np.sin(2.0 * x)
    b6 = np.cos(2.0 * x)
    
    # Basis 7-8: Second harmonics of v
    b7 = np.sin(2.0 * v)
    b8 = np.cos(2.0 * v)
    
    # Basis 9: Interaction term (sum frequency)
    b9 = np.sin(x + v)
    
    # Linear combination
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    return result
