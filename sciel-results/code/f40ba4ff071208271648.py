# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Gaussian RBF basis.
    
    We use 10 Gaussian RBFs in the (x, v) space. This provides a smooth, localized
    nonlinear basis that avoids the ill-conditioning of high-order global polynomials.
    The centers are fixed to a grid covering the input domain, and the parameters
    control the amplitude and width of each RBF.
    
    Basis:
    RBF_i(x, v) = exp(-||[x, v] - c_i||^2 / (2 * sigma^2))
    
    We have 10 parameters. We can use them as:
    - 5 centers (x_c, v_c) pairs? No, centers are usually fixed or learned via EM.
      Here params are linear coefficients in a fixed basis, or we can interpret
      params as amplitudes and widths?
      
    The interface says: "params, an array of length 10, with scipy.optimize.minimize... minimizing the training MSE."
    This implies the equation is linear in params? No, it just says minimize MSE.
    If the equation is nonlinear in params, BFGS can still work, but it's harder.
    Usually, "fitted constants" in these tasks implies a linear model where we fit weights,
    OR a nonlinear model where we fit all parameters.
    
    If we make the RBF widths fixed, the model is linear in the amplitudes.
    This is much easier for BFGS (or even least squares, though BFGS is used).
    Let's fix the centers and widths, and let params be the 10 amplitudes.
    
    Centers: A 2x5 grid or similar covering x in [-0.34, 0.36] and v in [-0.21, 0.17].
    Let's pick 10 distinct centers.
    x_centers = np.linspace(-0.35, 0.35, 5)
    v_centers = np.linspace(-0.2, 0.2, 2)
    Combine to 10 centers.
    
    Widths: Fixed sigma = 0.15 (approx half the range).
    
    This results in a linear model: y = sum(params[i] * RBF_i(x, v)).
    BFGS will converge quickly to the global minimum for this linear least squares problem
    (or close to it, as BFGS is a local method but the surface is convex for linear models).
    Actually, for linear models, the MSE surface is a convex quadratic. BFGS should find the global minimum.
    
    Wait, the previous score was very high (1.7e6). The inspirations had scores around 60.
    The current program (score 1.7e6) used a polynomial basis.
    The inspiration 2 (score 61) likely used a better basis or scaling.
    
    Let's try the RBF with fixed centers/widths and linear amplitudes.
    This is robust and well-conditioned if centers are spread out.
    """
    # Fixed centers for the 10 RBFs
    # x range: [-0.34, 0.36], v range: [-0.21, 0.17]
    # Let's define 10 centers manually to cover the space well.
    # 5 centers along x, 2 levels of v? Or 2x5 grid?
    # Let's do a 2x5 grid.
    x_centers = np.array([-0.30, -0.15, 0.0, 0.15, 0.30])
    v_centers = np.array([-0.10, 0.10])
    
    cx = np.repeat(x_centers, 2)
    cv = np.tile(v_centers, 5)
    
    # Fixed width
    sigma = 0.15
    
    # Compute RBF values for all 10 centers
    # dist_sq = (x - cx)^2 + (v - cv)^2
    # We need to do this for all rows of x, v and all 10 centers.
    # x, v are arrays of shape (N,).
    # cx, cv are arrays of shape (10,).
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (N, 10)
    dv = v[:, np.newaxis] - cv[np.newaxis, :]  # (N, 10)
    
    dist_sq = dx**2 + dv**2  # (N, 10)
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    # 2 * sigma^2 = 2 * 0.0225 = 0.045
    rbf_vals = np.exp(-dist_sq / 0.045)  # (N, 10)
    
    # Linear combination with params
    # params is shape (10,)
    # result = rbf_vals @ params
    result = rbf_vals @ params
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
