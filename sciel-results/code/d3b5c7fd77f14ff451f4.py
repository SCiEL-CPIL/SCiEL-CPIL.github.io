import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Spectral (Fourier) basis.
    
    Strategy:
    The target is likely the acceleration or a state derivative of a nonlinear oscillator.
    Oscillators exhibit periodic behavior. A polynomial basis (like the previous Duffing model)
    can suffer from ill-conditioning and poor convergence in BFGS when trying to fit 
    oscillatory data, especially if the "oscillation" is not a simple low-degree polynomial.
    
    A spectral basis using sin and cos functions provides:
    1. Bounded derivatives (crucial for BFGS stability).
    2. Natural representation of periodic dynamics.
    3. Fewer effective degrees of freedom needed to capture the "wave" shape.
    
    We use 10 parameters to weight a mix of low-frequency and mid-frequency harmonics
    of x and v. Since we don't have time 't', we use the state variables x and v as 
    proxies for phase. In a 2D phase space of an oscillator, trajectories are often 
    elliptical or spiral. Linear combinations of sin/cos of x and v can approximate 
    the vector field.
    
    Basis functions:
    1. sin(x)
    2. cos(x)
    3. sin(v)
    4. cos(v)
    5. sin(2*x)
    6. cos(2*x)
    7. sin(2*v)
    8. cos(2*v)
    9. sin(x + v)
    10. cos(x + v)
    
    This covers fundamental and second harmonic modes of the individual coordinates 
    and their sum, which is a reasonable inductive bias for coupled oscillator dynamics.
    """
    p = params
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Precompute trigonometric values for efficiency and clarity
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Higher harmonics
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_2v = np.sin(2 * v)
    cos_2v = np.cos(2 * v)
    
    # Cross term harmonic
    sin_xv = np.sin(x + v)
    cos_xv = np.cos(x + v)
    
    # Linear combination of the spectral basis
    # p0 * sin(x) + p1 * cos(x) + ...
    result = (
        p0 * sin_x +
        p1 * cos_x +
        p2 * sin_v +
        p3 * cos_v +
        p4 * sin_2x +
        p5 * cos_2x +
        p6 * sin_2v +
        p7 * cos_2v +
        p8 * sin_xv +
        p9 * cos_xv
    )
    
    return result
