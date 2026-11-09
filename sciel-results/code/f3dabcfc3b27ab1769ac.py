import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis for symbolic regression with 10 parameters.
    Covers: constant, linear, quadratic, cubic, exponential,
    logarithmic, trigonometric, rational, and square-root forms.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is positive for log/sqrt (domain is [0.1, 6] so this is safe,
    # but we clip for numerical safety during optimization)
    x_safe = np.clip(x, 1e-6, None)
    
    # Basis functions
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: exp(x)
    # 5: exp(-x)
    # 6: log(x)
    # 7: sin(x)
    # 8: cos(x)
    # 9: 1/x
    
    y = np.zeros_like(x)
    
    y += params[0] * 1.0
    y += params[1] * x
    y += params[2] * x**2
    y += params[3] * x**3
    
    # Exponential terms - clip to avoid overflow during optimization
    exp_x = np.exp(np.clip(x, -50, 50))
    exp_neg_x = np.exp(np.clip(-x, -50, 50))
    y += params[4] * exp_x
    y += params[5] * exp_neg_x
    
    y += params[6] * np.log(x_safe)
    y += params[7] * np.sin(x)
    y += params[8] * np.cos(x)
    y += params[9] * (1.0 / x_safe)
    
    return y
