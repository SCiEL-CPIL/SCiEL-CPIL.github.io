import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to centered, scaled ranges based on known training data
    # b: [1, 4.998] -> center ~3.0, scale 2.0
    # s: [1, 4.999] -> center ~3.0, scale 2.0
    # temp: [1.002, 39.98] -> center ~20.5, scale 19.5
    # pH: [2, 12] -> center 7.0, scale 5.0
    
    b_c = b - 3.0
    s_c = s - 3.0
    t_c = temp - 20.5
    p_c = pH - 7.0
    
    # Scale to roughly [-1, 1]
    b_n = b_c / 2.0
    s_n = s_c / 2.0
    t_n = t_c / 19.5
    p_n = p_c / 5.0
    
    # Quadratic terms
    b2 = b_n * b_n
    s2 = s_n * s_n
    t2 = t_n * t_n
    p2 = p_n * p_n
    
    # Interactions
    bt = b_n * t_n
    sp = s_n * p_n
    
    # Construct linear combination of basis functions
    # 10 parameters:
    # p[0]: intercept
    # p[1]-p[4]: linear terms (b, s, t, p)
    # p[5]-p[8]: quadratic terms (b^2, s^2, t^2, p^2)
    # p[9]: interaction term (b*t)
    
    z = (p[0] + 
         p[1] * b_n + 
         p[2] * s_n + 
         p[3] * t_n + 
         p[4] * p_n + 
         p[5] * b2 + 
         p[6] * s2 + 
         p[7] * t2 + 
         p[8] * p2 + 
         p[9] * bt)
    
    # Softplus activation to ensure smooth, non-negative output
    # For large positive z, softplus(z) ~ z
    # For large negative z, softplus(z) ~ exp(z) ~ 0
    # This provides a smooth approximation to ReLU and handles the non-negativity constraint
    
    return np.log1p(np.exp(z))
