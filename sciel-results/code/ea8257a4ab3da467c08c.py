import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] range based on known training ranges
    bn = (b - 1.0) / 4.0
    sn = s / 40.0
    tn = (temp - 1.0) / 39.0
    pn = (pH - 1.0) / 11.0
    
    # Fixed basis functions for each variable
    # Use cubic polynomials: 1, x, x^2, x^3 for each variable
    # Total: 4 bases per variable * 4 variables = 16, but we only have 10 params
    # So use: intercept + 2 for b + 2 for s + 2 for temp + 2 for pH + 1 extra = 10
    
    # For each variable, use: 1, x, x^2 (quadratic)
    # b: 1, bn, bn^2
    # s: 1, sn, sn^2
    # temp: 1, tn, tn^2
    # pH: 1, pn, pn^2
    # That's 12 terms, but we share the intercept.
    # Let's use: intercept + bn + bn^2 + sn + sn^2 + tn + tn^2 + pn + pn^2 + bn*sn
    # = 10 terms total
    
    # Compute basis values
    b1 = np.ones_like(bn)
    b2 = bn
    b3 = bn * bn
    
    s1 = np.ones_like(sn)
    s2 = sn
    s3 = sn * sn
    
    t1 = np.ones_like(tn)
    t2 = tn
    t3 = tn * tn
    
    p1 = np.ones_like(pn)
    p2 = pn
    p3 = pn * pn
    
    # Interaction term
    inter = bn * sn
    
    # Linear combination with 10 parameters
    # p[0]: intercept
    # p[1]: b linear
    # p[2]: b quadratic
    # p[3]: s linear
    # p[4]: s quadratic
    # p[5]: temp linear
    # p[6]: temp quadratic
    # p[7]: pH linear
    # p[8]: pH quadratic
    # p[9]: b*s interaction
    
    result = (p[0] * b1 + 
              p[1] * b2 + 
              p[2] * b3 + 
              p[3] * s2 + 
              p[4] * s3 + 
              p[5] * t2 + 
              p[6] * t3 + 
              p[7] * p2 + 
              p[8] * p3 + 
              p[9] * inter)
    
    return result
