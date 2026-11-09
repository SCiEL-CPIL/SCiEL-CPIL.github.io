import numpy as np

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model using a Generalized Additive Model (GAM).
    
    Strategy:
    Instead of a global quadratic surface with interactions, we use a sum of 
    univariate piecewise-linear basis functions (hats) for each input.
    This decouples the complexity, making the loss landscape smoother and 
    easier for BFGS to navigate. The 10 parameters are the coefficients 
    for these basis functions.
    
    Inputs:
    b: Population density [1, 5]
    s: Substrate concentration [1, 5]
    temp: Temperature [1, 40]
    pH: pH level [2, 12]
    
    Basis Functions:
    We define 3 hat functions per variable (centers at 25%, 50%, 75% of range).
    Total 12 basis functions. We use 10 parameters, mapping them to:
    p[0]: Intercept (bias)
    p[1]-p[9]: Coefficients for 9 selected hat functions to cover the space.
    
    Specific allocation:
    b: 2 hats (centers 2.5, 3.5)
    s: 2 hats (centers 2.5, 3.5)
    temp: 3 hats (centers 10, 20, 30)
    pH: 2 hats (centers 4.5, 9.5)
    Total 9 hats + 1 intercept = 10 params.
    
    Hat function definition:
    A hat centered at c with width w is:
    max(0, 1 - |x - c|/w)
    This creates a triangular bump.
    """
    
    # Input ranges and centers for hat functions
    # b: [1, 5]
    b1 = np.maximum(0, 1 - np.abs(b - 2.5) / 1.5)
    b2 = np.maximum(0, 1 - np.abs(b - 3.5) / 1.5)
    
    # s: [1, 5]
    s1 = np.maximum(0, 1 - np.abs(s - 2.5) / 1.5)
    s2 = np.maximum(0, 1 - np.abs(s - 3.5) / 1.5)
    
    # temp: [1, 40]
    t1 = np.maximum(0, 1 - np.abs(temp - 10) / 10)
    t2 = np.maximum(0, 1 - np.abs(temp - 20) / 10)
    t3 = np.maximum(0, 1 - np.abs(temp - 30) / 10)
    
    # pH: [2, 12]
    p1 = np.maximum(0, 1 - np.abs(pH - 4.5) / 2.5)
    p2 = np.maximum(0, 1 - np.abs(pH - 9.5) / 2.5)
    
    # Construct linear combination
    # p[0]: intercept
    # p[1], p[2]: b hats
    # p[3], p[4]: s hats
    # p[5], p[6], p[7]: temp hats
    # p[8], p[9]: pH hats
    
    z = (params[0] + 
         params[1] * b1 + params[2] * b2 + 
         params[3] * s1 + params[4] * s2 + 
         params[5] * t1 + params[6] * t2 + params[7] * t3 + 
         params[8] * p1 + params[9] * p2)
         
    # Use softplus to ensure smooth, non-negative output (growth rate >= 0)
    # This helps with the physical constraint and numerical stability
    return np.log1p(np.exp(z))
