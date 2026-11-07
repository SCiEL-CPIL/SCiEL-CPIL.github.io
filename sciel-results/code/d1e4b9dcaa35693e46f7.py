"""
Strategy: Improved Non-linear Parametric Model using a Separable Gaussian and a Linear Term.
The previous Sum of Isotropic Gaussians was prone to optimization difficulties (score 253).
This model uses a single anisotropic (separable) Gaussian basis function combined with a linear background.
This structure is more flexible for capturing both global trends (linear) and local non-linear features (Gaussian)
while maintaining a simpler optimization landscape than a sum of multiple independent Gaussians.

Parameters (10 total):
0: w   (Gaussian weight)
1: mx  (Gaussian center x)
2: my  (Gaussian center y)
3: mz  (Gaussian center z)
4: l_sx (Log-width for x)
5: l_sy (Log-width for y)
6: l_sz (Log-width for z)
7: ax  (Linear coeff x)
8: ay  (Linear coeff y)
9: az  (Linear coeff z)

Widths are computed as exp(log_width) to ensure positivity and smooth gradients.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    f(x,y,z) = w * exp(-0.5 * ((x-mx)^2/sx^2 + (y-my)^2/sy^2 + (z-mz)^2/sz^2)) + ax*x + ay*y + az*z
    """
    # Extract parameters
    w = params[0]
    mx = params[1]
    my = params[2]
    mz = params[3]
    
    log_sx = params[4]
    log_sy = params[5]
    log_sz = params[6]
    
    ax = params[7]
    ay = params[8]
    az = params[9]
    
    # Compute widths from log-space to ensure positivity
    # Clip to avoid extreme values that could lead to numerical instability in exp
    # exp(10) ~ 22000, exp(-10) ~ 4.5e-5
    # Given domain [0.5, 2], these are safe bounds for width.
    sx = np.exp(np.clip(log_sx, -10, 10))
    sy = np.exp(np.clip(log_sy, -10, 10))
    sz = np.exp(np.clip(log_sz, -10, 10))
    
    # Calculate differences
    dx = x - mx
    dy = y - my
    dz = z - mz
    
    # Calculate Gaussian term
    # Using (dx/sx)^2 is equivalent to dx^2 / sx^2 but slightly more efficient
    g_arg = -0.5 * ( (dx / sx)**2 + (dy / sy)**2 + (dz / sz)**2 )
    
    # Avoid overflow/underflow issues in exp by clipping the argument
    # If arg is very negative, exp is 0. If arg is near 0, exp is 1.
    # Max possible negative value: dx^2/sx^2 can be large if sx is small.
    # If sx = exp(-10) = 4.5e-5, dx=1.5, dx/sx = 33000. squared = 1e9.
    # exp(-1e9) is 0. This is fine.
    # We don't need to clip g_arg unless it's positive (which it isn't, due to -0.5 and squares).
    
    g_term = np.exp(g_arg)
    
    # Calculate linear term
    l_term = ax * x + ay * y + az * z
    
    return w * g_term + l_term
# EVOLVE-BLOCK-END
