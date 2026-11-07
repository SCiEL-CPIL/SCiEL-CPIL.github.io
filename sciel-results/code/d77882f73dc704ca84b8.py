import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)
    
    # Hybrid basis: Polynomial (Chebyshev) + Sinusoidal components
    # This allows capturing both smooth polynomial trends and periodic features.
    
    # 1. Chebyshev Polynomials (for smooth, non-periodic trends)
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x * x - 1
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y * y - 1
    
    # 2. Sinusoidal Basis (for periodic features)
    # Using frequencies that are common in test functions (pi, 2pi)
    # sin(pi*x), cos(pi*x), sin(pi*y), cos(pi*y)
    # sin(pi*x*y) or similar interactions? Let's stick to separable first.
    # We have 10 params total.
    # Let's allocate:
    # 4 for polynomial (1, x, y, x*y) -- wait, Chebyshev is better conditioned.
    # Let's use:
    # p0: 1
    # p1: x
    # p2: y
    # p3: x*y
    # p4: sin(pi*x)
    # p5: cos(pi*x)
    # p6: sin(pi*y)
    # p7: cos(pi*y)
    # p8: x*x (or T2_x)
    # p9: y*y (or T2_y)
    
    # Actually, mixing raw polynomials and Chebyshev can be redundant.
    # Let's stick to a clean set.
    # Basis:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x*y
    # 4: sin(pi*x)
    # 5: cos(pi*x)
    # 6: sin(pi*y)
    # 7: cos(pi*y)
    # 8: sin(pi*x*y) ? No, interaction terms can be tricky. 
    #    Let's try x^2 and y^2 instead, or higher freq sinusoids.
    #    Given the hint "sin(3x+4y)", simple separable sinusoids might not be enough if it's a sum.
    #    But sin(3x+4y) = sin(3x)cos(4y) + cos(3x)sin(4y).
    #    So we need frequencies 3 and 4? Or just generic frequencies?
    #    The optimizer will tune the amplitudes. If the basis has the right frequencies, it can fit.
    #    If we use pi, it fits sin(pi*x). If the target is sin(x), it's a different frequency.
    #    However, without knowing the frequency, we can't parametrize it easily with only 10 params.
    #    BUT, the insight from the previous run said: "Replaces fixed-frequency linear basis with adaptive-frequency parametric sinusoids... Measured: 0.2177 -> 2.718."
    #    Wait, the score went UP (worse) in that case? "0.2177 -> 2.718". Lower is better. So it got worse.
    #    The current score is 7.02. The previous best was 1.63.
    #    The current program is a regression from the 1.63 score program?
    #    The parent p29 had score 1.63. The child p42 (current) has score 7.02.
    #    So the current program is WORSE than the parent.
    #    The parent p29 had: "Basic features... Trigonometric features at multiple frequencies... sx1, cx1, sy1, cy1... sx2, cx2..."
    #    So the parent used sin(pi*x), cos(pi*x), sin(2pi*x)... etc.
    #    The current program switched to pure Chebyshev polynomials.
    #    The score dropped from 1.63 to 7.02.
    #    This implies the target function has significant non-polynomial (trigonometric) components that the pure polynomial basis cannot fit well.
    #    
    #    Strategy: Restore the hybrid approach but refine it.
    #    Keep some polynomial terms for the baseline/trend, and add sinusoidal terms for the periodic parts.
    #    
    #    Proposed Basis (10 terms):
    #    0: 1
    #    1: x
    #    2: y
    #    3: x*y
    #    4: sin(pi*x)
    #    5: cos(pi*x)
    #    6: sin(pi*y)
    #    7: cos(pi*y)
    #    8: sin(2*pi*x)
    #    9: sin(2*pi*y)
    
    # This covers the fundamental frequencies (pi) and the second harmonic (2pi) for both x and y.
    # It also includes the basic linear and interaction polynomial terms.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * y
    
    b4 = np.sin(np.pi * x)
    b5 = np.cos(np.pi * x)
    b6 = np.sin(np.pi * y)
    b7 = np.cos(np.pi * y)
    
    b8 = np.sin(2 * np.pi * x)
    b9 = np.sin(2 * np.pi * y)
    
    result = (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
              params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
              params[8]*b8 + params[9]*b9)
    
    return result
