import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using an anisotropic RBF (Gaussian kernel) basis.
    This non-parametric-like approach with fixed centers allows flexible local
    feature capture, differing structurally from global polynomial/tanh bases.
    
    Parameters (length 10):
    - params[0:5] : weights for 5 Gaussian kernels
    - params[5]   : inverse width for x (sigma_x^-2)
    - params[6]   : inverse width for y (sigma_y^-2)
    - params[7]   : linear bias for x
    - params[8]   : linear bias for y
    - params[9]   : constant bias
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Fixed centers: (0,0), (0.5,0), (-0.5,0), (0,0.5), (0,-0.5)
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5]
    ])
    
    # Extract parameters
    weights = params[0:5]
    inv_wx = params[5]
    inv_wy = params[6]
    bx = params[7]
    by = params[8]
    b0 = params[9]
    
    # Ensure widths are positive to avoid numerical issues
    # If inv_w is negative, the Gaussian becomes an anti-Gaussian (grows with distance),
    # which is not desired. We clamp to a small positive value.
    # However, BFGS might explore negative values. 
    # A safer approach: use exp(-inv_w * dist^2). If inv_w is negative, it grows.
    # To keep it stable, we can use max(inv_w, 0.1) or similar.
    # But that makes the function non-smooth.
    # Alternative: use 1.0 / (1.0 + inv_w*dist^2) ? No, let's stick to Gaussian but ensure stability.
    # Actually, if the optimizer finds a negative width, the prediction might explode.
    # Let's use a softplus-like or just clamp in the calculation for safety?
    # The task says "non-finite error costs 1e30". So we just need to avoid inf/nan.
    # If inv_w is very large positive, exp(-large) -> 0. Safe.
    # If inv_w is negative, exp(positive) -> inf. Unsafe.
    # Let's clamp inv_wx and inv_wy to be at least 0.0? 
    # If 0, it becomes a constant kernel (1). 
    # Let's clamp to a small positive number to ensure decay.
    inv_wx_safe = np.maximum(inv_wx, 0.0)
    inv_wy_safe = np.maximum(inv_wy, 0.0)
    
    # Compute RBF contributions
    pred = np.zeros_like(x)
    
    for i in range(5):
        cx = centers[i, 0]
        cy = centers[i, 1]
        dx = x - cx
        dy = y - cy
        dist2 = inv_wx_safe * dx * dx + inv_wy_safe * dy * dy
        # Clip dist2 to prevent overflow in exp if inv_w is huge and dx is large (though dx is bounded [-1,1])
        # Max dist2 = 1*1 + 1*1 = 2 if inv_w=1. If inv_w=1000, dist2=2000. exp(-2000) is 0.
        # If inv_w is negative, we clamped it to 0, so dist2=0, exp(0)=1.
        # So it's safe.
        kernel = np.exp(-dist2)
        pred += weights[i] * kernel
    
    # Add linear and constant biases
    pred += bx * x
    pred += by * y
    pred += b0
    
    return pred
