import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Duffing oscillator model 
    with Gaussian RBF corrections.
    
    Parameters:
      p0: Global bias
      p1: Natural frequency squared (omega_0^2)
      p2: Nonlinearity coefficient (beta)
      p3: Damping coefficient (gamma)
      p4: Forcing amplitude (F_0)
      p5: Forcing frequency (Omega)
      p6: Forcing phase (phi)
      p7: RBF weight 1
      p8: RBF weight 2
      p9: RBF weight 3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Duffing Oscillator Terms
    # Equation: a = -omega^2 * x - beta * x^3 + gamma * v + F0 * cos(Omega * t + phi)
    # Note: Signs are chosen to allow the optimizer to find the correct direction, 
    # but standard Duffing is x'' + gamma x' + omega^2 x + beta x^3 = F(t)
    # So x'' = -gamma x' - omega^2 x - beta x^3 + F(t)
    
    term_linear = -p1 * x
    term_nonlinear = -p2 * (x ** 3)
    term_damping = -p3 * v
    term_forcing = p4 * np.cos(p5 * t + p6)
    
    base_prediction = p0 + term_linear + term_nonlinear + term_damping + term_forcing
    
    # RBF Corrections for residual errors
    # Define 3 fixed centers in the (x, v) space (t is handled by forcing term mostly)
    # x range: [-0.34, 0.36], v range: [-0.21, 0.17]
    
    # Center 1: Near origin
    cx1, cv1 = 0.0, 0.0
    # Center 2: Positive x, low v
    cx2, cv2 = 0.2, -0.1
    # Center 3: Negative x, high v
    cx3, cv3 = -0.2, 0.1
    
    # Widths
    w_x = 0.25
    w_v = 0.15
    
    # Compute RBFs
    rbf1 = np.exp(-0.5 * (((x - cx1) / w_x) ** 2 + ((v - cv1) / w_v) ** 2))
    rbf2 = np.exp(-0.5 * (((x - cx2) / w_x) ** 2 + ((v - cv2) / w_v) ** 2))
    rbf3 = np.exp(-0.5 * (((x - cx3) / w_x) ** 2 + ((v - cv3) / w_v) ** 2))
    
    result = base_prediction + p7 * rbf1 + p8 * rbf2 + p9 * rbf3
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
    
    return result
