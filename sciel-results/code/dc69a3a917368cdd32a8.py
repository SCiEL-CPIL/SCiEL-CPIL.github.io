"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric Nonlinear Oscillator Model (Van der Pol + Duffing).
    
    Strategy:
    - Instead of a local basis (RBF) or generic polynomial, use a physics-inspired structure.
    - The model assumes the system is a self-sustained oscillator with a limit cycle.
    - The dynamics are governed by:
      y = a0 + a1*x + a2*x^3 + b1*v + b2*v^3 + c1*x*v + c2*x^2*v + c3*v^2 + c4*x^2 + c5*x*v^2 + c6*v^4
    - This structure explicitly captures:
      - Linear and cubic stiffness (Duffing terms: a1*x, a2*x^3)
      - Linear and cubic damping (Van der Pol terms: b1*v, b2*v^3)
      - Cross-coupling terms (c1*x*v, etc.) which may represent parametric excitation or
        state-dependent damping/stiffness interactions.
    - Time 't' is excluded from the functional form, assuming the data lies on an autonomous
      trajectory where the state (x, v) uniquely determines the acceleration y. This reduces
      parameter count and avoids fitting noise in the time dimension, focusing BFGS on the
      physical coefficients.
    - Inputs x and v are normalized to improve numerical conditioning.
      x_norm = (x - mean_x) / scale_x
      v_norm = (v - mean_v) / scale_v
    - The 10 parameters correspond to the coefficients of the basis functions.
      Starting from params=1, the model is a reasonable initial guess for a nonlinear system.
      
    Parameters (length 10):
    p[0] -> a0 (offset)
    p[1] -> a1 (linear stiffness)
    p[2] -> a2 (cubic stiffness)
    p[3] -> b1 (linear damping)
    p[4] -> b2 (cubic damping)
    p[5] -> c1 (x*v coupling)
    p[6] -> c2 (x^2*v coupling)
    p[7] -> c3 (v^2 coupling)
    p[8] -> c4 (x^2 coupling)
    p[9] -> c5 (x*v^2 coupling)
    """
    p = params
    
    # Normalization constants based on training ranges:
    # x in [-0.2056, 0.1749] -> mean ~ -0.015, range ~ 0.38
    # v in [-0.3391, 0.3624] -> mean ~ 0.011, range ~ 0.70
    # We normalize to roughly [-1, 1] to make the coefficients interpretable and BFGS stable.
    
    mean_x = -0.01535
    scale_x = 0.19
    mean_v = 0.01165
    scale_v = 0.35
    
    x_norm = (x - mean_x) / scale_x
    v_norm = (v - mean_v) / scale_v
    
    # Basis functions
    # We use a mix of low-order terms to capture the essential nonlinearity.
    # The Van der Pol equation is: x'' - mu(1-x^2)x' + x = 0 -> x'' = -x + mu x' - mu x^2 x'
    # The Duffing equation is: x'' + delta x' + alpha x + beta x^3 = F0 cos(omega t)
    # Combining these and adding cross terms for generality:
    
    term_x = x_norm
    term_x3 = x_norm**3
    term_v = v_norm
    term_v3 = v_norm**3
    term_xv = x_norm * v_norm
    term_x2v = x_norm**2 * v_norm
    term_v2 = v_norm**2
    term_x2 = x_norm**2
    term_xv2 = x_norm * v_norm**2
    # term_v4 = v_norm**4 (dropped to keep param count at 10, or could replace one of the above if needed)
    
    y = (p[0] + 
         p[1] * term_x + 
         p[2] * term_x3 + 
         p[3] * term_v + 
         p[4] * term_v3 + 
         p[5] * term_xv + 
         p[6] * term_x2v + 
         p[7] * term_v2 + 
         p[8] * term_x2 + 
         p[9] * term_xv2)
         
    return y

# EVOLVE-BLOCK-END
