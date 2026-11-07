import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fully parametric sine-based feature map.
    
    Strategy:
    Instead of using fixed frequencies (which restricts the model to a specific
    harmonic structure), we make the frequencies and phases learnable. This allows
    the optimizer (BFGS) to adjust the basis functions to match the specific
    periodicity, scaling, and phase of the hidden target function.
    
    The model is:
    y(x) = c0 + sum_{i=1}^{3} a_i * sin(f_i * x + p_i)
    
    Since we have 10 parameters total, we can afford 3 such terms (3 params each: a, f, p)
    plus 1 bias term. This provides significant flexibility in the frequency domain.
    
    Parameters (length 10):
    params[0]: Constant term (bias)
    params[1], params[2], params[3]: Amplitude, frequency, phase for term 1
    params[4], params[5], params[6]: Amplitude, frequency, phase for term 2
    params[7], params[8], params[9]: Amplitude, frequency, phase for term 3
    """
    c0 = params[0]
    
    # Extract parameters for the 3 sine terms
    # a1, f1, p1
    a1, f1, p1 = params[1], params[2], params[3]
    # a2, f2, p2
    a2, f2, p2 = params[4], params[5], params[6]
    # a3, f3, p3
    a3, f3, p3 = params[7], params[8], params[9]
    
    # Compute the sine terms
    # Ensure numerical stability by clipping frequencies if they get too large?
    # BFGS starting from 1.0 usually keeps things reasonable, but let's be safe.
    # However, standard sine is fine for reasonable frequencies.
    
    term1 = a1 * np.sin(f1 * x + p1)
    term2 = a2 * np.sin(f2 * x + p2)
    term3 = a3 * np.sin(f3 * x + p3)
    
    y = c0 + term1 + term2 + term3
    
    return y
