import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration using physically-motivated terms with tanh saturation.
    This structure improves BFGS convergence by providing a better-conditioned
    parameter landscape compared to raw polynomials or Chebyshev expansions.
    
    Inputs x and v are normalized to O(1) range.
    Terms include linear/nonlinear restoring forces, damping, and cross-couplings.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] based on training data ranges
    # x range: [-0.5728, 0.5275] -> mid: -0.02265, half_range: 0.55015
    # v range: [-0.1754, 0.2821] -> mid: 0.05335, half_range: 0.22875
    
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # Precompute terms
    tanh_xn = np.tanh(xn)
    tanh_vn = np.tanh(vn)
    
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Construct the model
    # p0: bias
    # p1: linear restoring
    # p2: nonlinear restoring (tanh)
    # p3: cubic restoring (Duffing)
    # p4: linear damping
    # p5: nonlinear damping (tanh)
    # p6: cubic damping
    # p7: linear coupling
    # p8: quadratic-cubic coupling (x^2 v)
    # p9: linear-quadratic coupling (x v^2)
    
    return (p0 +
            p1 * xn +
            p2 * tanh_xn +
            p3 * xn3 +
            p4 * vn +
            p5 * tanh_vn +
            p6 * vn3 +
            p7 * xn * vn +
            p8 * xn2 * vn +
            p9 * xn * vn2)
