"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate target function using a Fourier series basis on [-2, 2].
    Uses 10 parameters: 5 cosine terms and 5 sine terms.
    Basis functions: cos(2*pi*k*x/4) and sin(2*pi*k*x/4) for k=0..4.
    Since x in [-2, 2], let u = pi*x/2, so u in [-pi/2, pi/2].
    Alternatively, map x to t in [0, 1] via t = (x+2)/4.
    Then basis is cos(2*pi*k*t) and sin(2*pi*k*t) for k=0..4.
    This ensures periodicity on the domain and orthogonality.
    
    params[0] * cos(0) + params[1] * sin(0) + ... is not quite right.
    Let's define:
    k=0: cos(0) = 1, sin(0) = 0. So params[0] is the constant term.
    k=1: cos(2*pi*t), sin(2*pi*t)
    k=2: cos(4*pi*t), sin(4*pi*t)
    k=3: cos(6*pi*t), sin(6*pi*t)
    k=4: cos(8*pi*t), sin(8*pi*t)
    
    Total 1 + 2*4 = 9 terms? No, 10 params.
    Let's use k=0..4 for cos (5 terms) and k=0..4 for sin (5 terms).
    But sin(0) is always 0, so params[1] would be unused if we pair them that way.
    Better:
    params[0]: DC term (cos(0))
    params[1]: cos(2*pi*t)
    params[2]: sin(2*pi*t)
    params[3]: cos(4*pi*t)
    params[4]: sin(4*pi*t)
    params[5]: cos(6*pi*t)
    params[6]: sin(6*pi*t)
    params[7]: cos(8*pi*t)
    params[8]: sin(8*pi*t)
    params[9]: cos(10*pi*t) -- Wait, that's 10 terms.
    
    Let's structure it as:
    result = params[0]
    for k in range(1, 5):
        result += params[2*k-1] * cos(2*pi*k*t)
        result += params[2*k] * sin(2*pi*k*t)
    This uses params[0] to params[8]. That's 9 params.
    We have 10 params.
    Let's add one more frequency: k=5.
    result = params[0]
    for k in range(1, 6):
        result += params[2*k-1] * cos(2*pi*k*t)
        result += params[2*k] * sin(2*pi*k*t)
    This uses params[0] to params[10]. That's 11 params. Too many.
    
    Let's just use 5 cos and 5 sin terms, ignoring the fact that sin(0)=0.
    Or, simpler:
    t = (x + 2) / 4  # maps [-2, 2] to [0, 1]
    
    result = np.zeros_like(x)
    for k in range(5):
        angle = 2 * np.pi * k * t
        # Use params[2*k] for cos, params[2*k+1] for sin
        # But for k=0, sin(0)=0, so params[1] is wasted.
        # To utilize all 10 params effectively, let's shift the phase or frequencies.
        
    Alternative: Use frequencies 0, 1, 2, 3, 4, 5, 6, 7, 8, 9? No, that's 10 cos terms.
    Let's use 5 cos and 5 sin terms with frequencies 1, 2, 3, 4, 5.
    params[0]: cos(2*pi*1*t)
    params[1]: sin(2*pi*1*t)
    params[2]: cos(2*pi*2*t)
    params[3]: sin(2*pi*2*t)
    params[4]: cos(2*pi*3*t)
    params[5]: sin(2*pi*3*t)
    params[6]: cos(2*pi*4*t)
    params[7]: sin(2*pi*4*t)
    params[8]: cos(2*pi*5*t)
    params[9]: sin(2*pi*5*t)
    
    This misses the DC term (constant). A constant term is crucial.
    So, let's drop the highest frequency sine or cosine?
    Or, use frequencies 0, 1, 2, 3, 4 for cos and 1, 2, 3, 4, 5 for sin?
    
    Let's try:
    params[0]: Constant (1)
    params[1]: cos(2*pi*1*t)
    params[2]: sin(2*pi*1*t)
    params[3]: cos(2*pi*2*t)
    params[4]: sin(2*pi*2*t)
    params[5]: cos(2*pi*3*t)
    params[6]: sin(2*pi*3*t)
    params[7]: cos(2*pi*4*t)
    params[8]: sin(2*pi*4*t)
    params[9]: cos(2*pi*5*t)  -- No sin for 5?
    
    This is unbalanced.
    
    Let's go with a simpler, more robust approach:
    Map x to u in [-1, 1] by u = x/2.
    Use Fourier series in u:
    sum_{k=0}^4 [ a_k cos(k*pi*u) + b_k sin(k*pi*u) ]
    Here k=0..4 gives 5 cos and 5 sin terms. Total 10 params.
    cos(0) = 1, sin(0) = 0.
    So params[0] is the DC term.
    params[1] is b_0 (multiplies sin(0)=0, so it's dead).
    This is inefficient.
    
    Let's use:
    a_0, a_1, a_2, a_3, a_4 (5 cosines)
    b_1, b_2, b_3, b_4, b_5 (5 sines)
    
    u = x / 2.0
    result = params[0] * 1.0  # a_0 * cos(0)
    for k in range(1, 5):
        result += params[k] * np.cos(k * np.pi * u)
    for k in range(1, 6):
        result += params[4 + k] * np.sin(k * np.pi * u)
        
    This uses params[0] to params[9].
    params[0]: a_0
    params[1]: a_1
    params[2]: a_2
    params[3]: a_3
    params[4]: a_4
    params[5]: b_1
    params[6]: b_2
    params[7]: b_3
    params[8]: b_4
    params[9]: b_5
    
    This looks good. It covers frequencies 0-4 for cos and 1-5 for sin.
    """
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
        
    N = x.shape[0]
    if N == 0:
        return np.array([])
        
    u = x / 2.0  # Map [-2, 2] to [-1, 1]
    
    result = params[0] * np.ones(N)  # DC term
    
    # Cosine terms k=1 to 4
    # params[1] to params[4]
    for k in range(1, 5):
        result += params[k] * np.cos(k * np.pi * u)
        
    # Sine terms k=1 to 5
    # params[5] to params[9]
    for k in range(1, 6):
        result += params[4 + k] * np.sin(k * np.pi * u)
        
    return result
# EVOLVE-BLOCK-END
