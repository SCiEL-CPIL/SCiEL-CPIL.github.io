# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 diverse basis functions.
    
    The model is:
    f(x) = w0*1 + w1*x + w2*x^2 + w3*x^3 + w4*x^4 
           + w5*sin(x) + w6*cos(x) + w7*sin(2x) + w8*cos(2x) + w9*exp(-x^2/2)
    
    This model is linear in the parameters (weights), making the optimization
    problem convex. BFGS will converge to the global least-squares solution
    efficiently, avoiding the local minima traps of non-linear models like
    RBFs with learnable widths.
    
    The basis includes:
    - Polynomials up to degree 4 to capture smooth trends.
    - Sine and cosine terms (frequencies 1 and 2) to capture periodic behavior.
    - A Gaussian bump at x=0 to capture localized non-polynomial features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract parameters (weights)
    # Ensure params is 1D and length 10
    w = params[:10]
    
    # Build the basis matrix Phi of shape (N, 10)
    # Phi[:, j] = phi_j(x)
    
    # 1. Constant
    phi0 = np.ones_like(x)
    
    # 2-5. Polynomials
    phi1 = x
    phi2 = x**2
    phi3 = x**3
    phi4 = x**4
    
    # 6-9. Trigonometric
    phi5 = np.sin(x)
    phi6 = np.cos(x)
    phi7 = np.sin(2.0 * x)
    phi8 = np.cos(2.0 * x)
    
    # 10. Gaussian bump
    phi9 = np.exp(-0.5 * x**2)
    
    # Stack into matrix
    # Using column_stack or hstack
    Phi = np.column_stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9))
    
    # Compute prediction: y = Phi @ w
    # Ensure w is finite
    w = np.where(np.isfinite(w), w, 0.0)
    
    result = Phi @ w
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
