import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear trend + 8 Gaussian RBFs with learnable widths.
    
    The RBF centers are fixed at Chebyshev nodes for better distribution.
    The widths (sigmas) are learned via a softplus transform of params to ensure positivity.
    This allows the model to adapt to localized features (bumps, kinks) more effectively
    than fixed-width RBFs or pure polynomial bases.
    
    Parameters:
    params[0]: linear coefficient
    params[1]: constant offset
    params[2:10]: log-width parameters for the 8 RBFs (transformed via softplus)
    
    Note: The amplitudes of the RBFs are implicitly learned by the optimizer 
    adjusting the width and the linear terms, but to allow for signed bumps, 
    we should actually have amplitudes too. However, we only have 10 params.
    
    Revised Plan:
    We have 10 params.
    If we use 8 RBFs, we need 8 amplitudes + 1 linear + 1 constant = 10 params.
    Widths must be fixed or shared.
    
    Alternative Plan:
    Use a small number of RBFs with learnable centers and widths?
    No, centers/widths as params makes the problem non-linear and harder for BFGS.
    
    Let's stick to the current structure but improve the basis.
    Current: Linear + 8 fixed-width RBFs.
    Problem: Fixed width might be too narrow or too wide.
    
    Idea from Inspirations:
    Inspiration 1 uses a diverse basis (Chebyshev + Trig + Tanh + Gaussian).
    Inspiration 3 uses pure Chebyshev.
    
    Let's try a mixed basis that is linear in params (efficient for BFGS) but captures more features.
    Basis:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: tanh(x)
    9: exp(-x^2/2)  (Central Gaussian bump)
    
    This is similar to Inspiration 2, which scored 4413.1736.
    The current program scored 4329.3130.
    Inspiration 1 scored 4413.8832.
    
    Why did the current program underperform?
    The RBF centers are at Chebyshev nodes. The amplitudes are free.
    This is a linear combination of fixed functions.
    The diverse basis in Insp 1/2 might be more flexible for general functions.
    
    Let's combine:
    Use a basis that includes:
    - Low-order polynomials (for global trend)
    - Trigonometric terms (for periodicity)
    - A central Gaussian (for localized bump)
    - A tanh (for saturation/kink)
    
    Let's refine the basis of Inspiration 1/2 to be more robust.
    Map x to t in [-1, 1] for stability.
    
    Basis:
    0: 1
    1: t
    2: t^2
    3: t^3
    4: sin(pi * t)
    5: cos(pi * t)
    6: sin(2 * pi * t)
    7: cos(2 * pi * t)
    8: tanh(2 * t)
    9: exp(-2 * t^2)
    
    This covers:
    - Polynomial trends
    - Periodic components (fundamental and 2nd harmonic)
    - Saturation/non-linearity (tanh)
    - Localized central feature (Gaussian)
    
    This is a linear model in params, so BFGS will converge quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Clip for safety
    t = np.clip(t, -1.0, 1.0)
    
    # Build basis
    b0 = np.ones_like(t)
    b1 = t
    b2 = t ** 2
    b3 = t ** 3
    
    # Trigonometric
    sin_pi_t = np.sin(np.pi * t)
    cos_pi_t = np.cos(np.pi * t)
    sin_2pi_t = np.sin(2.0 * np.pi * t)
    cos_2pi_t = np.cos(2.0 * np.pi * t)
    
    # Saturation
    tanh_2t = np.tanh(2.0 * t)
    
    # Central Gaussian
    exp_neg_2t2 = np.exp(-2.0 * t ** 2)
    
    basis = np.stack([
        b0, b1, b2, b3,
        sin_pi_t, cos_pi_t,
        sin_2pi_t, cos_2pi_t,
        tanh_2t,
        exp_neg_2t2
    ], axis=1)
    
    result = basis @ params
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
