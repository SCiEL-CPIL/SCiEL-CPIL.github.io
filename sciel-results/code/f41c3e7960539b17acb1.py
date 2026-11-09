import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model: sum of 5 adaptive power terms.
    Each term is p_{2i} * (x + small_offset)^{p_{2i+1}} * (y + small_offset)^{p_{2i+2}} ... 
    Wait, we have 10 params. Let's use 5 terms.
    Term i: c_i * (x)^{e_{i,x}} * (y)^{e_{i,y}}
    Params: 
    p[0]=c0, p[1]=ex0, p[2]=ey0
    p[3]=c1, p[4]=ex1, p[5]=ey1
    p[6]=c2, p[7]=ex2, p[8]=ey2
    p[9]=c3 (and maybe we need one more exponent? Or use a fixed structure for the 5th?)
    
    Let's refine: 10 params.
    5 terms, each with 2 params? c_i and exponent?
    Or 4 terms with 2.5 params? No.
    
    Let's try: 5 terms.
    Term 1: p[0] * x^(p[1])
    Term 2: p[2] * y^(p[3])
    Term 3: p[4] * (x*y)^(p[5])
    Term 4: p[6] * (x+y)^(p[7])
    Term 5: p[8] * (x^2 + y^2)^(p[9]/2)
    
    This is highly nonlinear in params. BFGS might struggle with exponents if base is negative or zero.
    x, y in [-1, 1]. 
    x^(exponent) is problematic for x<0 if exponent is not integer.
    We need to ensure bases are positive or handle signs.
    Since x, y can be negative, x^a is undefined for non-integer a in real numbers.
    
    Alternative: Use sin/cos with adaptive frequencies?
    p[0] + p[1]*sin(p[2]*x) + p[3]*cos(p[4]*y) + p[5]*sin(p[6]*x*p[7]*y) ...
    This is safe for all real inputs.
    
    Let's try a sum of sines with adaptive frequencies and amplitudes.
    10 params:
    1. p[0]: constant offset
    2. p[1]: amplitude 1
    3. p[2]: frequency 1 (for x)
    4. p[3]: amplitude 2
    5. p[4]: frequency 2 (for y)
    6. p[5]: amplitude 3
    7. p[6]: frequency 3 (for x+y)
    8. p[7]: amplitude 4
    9. p[8]: frequency 4 (for x-y)
    10. p[9]: amplitude 5 (for x*y)
    
    This is a Fourier-like series with adaptive frequencies.
    BFGS can handle this if we initialize frequencies to small non-zero values to avoid zero gradients?
    Starting from all ones: freq=1, amp=1.
    
    Let's check stability.
    f(x,y) = p[0] + p[1]*sin(p[2]*x) + p[3]*cos(p[4]*y) + p[5]*sin(p[6]*(x+y)) + p[7]*cos(p[8]*(x-y)) + p[9]*sin(p[2]*x*p[4]*y) ??
    Mixing params might cause coupling issues.
    
    Let's keep it simpler:
    p[0]: offset
    p[1]: amp_x, p[2]: freq_x
    p[3]: amp_y, p[4]: freq_y
    p[5]: amp_xy, p[6]: freq_xy (applied to x*y)
    p[7]: amp_sum, p[8]: freq_sum (applied to x+y)
    p[9]: amp_diff, (reuse freq_sum? or fixed freq?)
    
    Let's do:
    term1 = p[1] * sin(p[2] * x)
    term2 = p[3] * cos(p[4] * y)
    term3 = p[5] * sin(p[6] * x * y)
    term4 = p[7] * cos(p[8] * (x + y))
    term5 = p[9] * sin(x * y)  # Fixed freq 1 for interaction, amplitude p[9]
    
    This uses all 10 params.
    p[0] is constant.
    
    This model is smooth, bounded (unless amplitudes grow huge, but MSE will penalize that), and captures periodic/interaction effects.
    It is fundamentally different from the linear basis.
    """
    p = params
    
    # Ensure numerical stability: clip parameters to prevent overflow in sin/cos arguments?
    # sin/cos are bounded, so large frequencies just cause rapid oscillation.
    # Amplitudes can grow, but MSE minimization will keep them reasonable.
    
    t0 = p[0]
    
    # Adaptive frequency sine for x
    t1 = p[1] * np.sin(p[2] * x)
    
    # Adaptive frequency cosine for y
    t2 = p[3] * np.cos(p[4] * y)
    
    # Adaptive frequency sine for interaction x*y
    t3 = p[5] * np.sin(p[6] * x * y)
    
    # Adaptive frequency cosine for sum x+y
    t4 = p[7] * np.cos(p[8] * (x + y))
    
    # Fixed frequency sine for interaction, adaptive amplitude
    # Using p[9] as amplitude
    t5 = p[9] * np.sin(x * y)
    
    return t0 + t1 + t2 + t3 + t4 + t5
