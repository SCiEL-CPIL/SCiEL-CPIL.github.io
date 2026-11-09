<<<<<<< SEARCH
    # Extract parameters
    mu_max = softplus(p[0])
    K_s = softplus(p[1])
    T_opt = p[2]
    T_width = softplus(p[3])
    pH_opt = p[4]
    pH_width = softplus(p[5])
    offset = p[6]
    b_lin = p[7]
    b_quad = p[8]
    s_inh = softplus(p[9])
    
    # Normalize inputs based on known ranges
    bn = (b - 1.0) / 4.0
    tn = (temp - 30.0) / 10.0
    pn = (pH - 7.0) / 3.0
    
    # Monod equation for substrate limitation
    sfac = s / (K_s + s + 1e-9)
    
    # Haldane term for substrate inhibition (optional, controlled by s_inh)
    # This can capture non-monotonic behavior if s_inh is significant
    haldane = 1.0 / (1.0 + s_inh * s / (K_s + 1e-9))
    sfac *= haldane
    
    # Temperature bell curve
    tfac = np.exp(-0.5 * ((tn - T_opt) / (T_width + 1e-9))**2)
    
    # pH bell curve
    pfac = np.exp(-0.5 * ((pn - pH_opt) / (pH_width + 1e-9))**2)
    
    # Density modulation (linear and quadratic)
    dfac = 1.0 + b_lin * bn + b_quad * bn * bn
    
    # Combine all factors
    return mu_max * sfac * tfac * pfac * dfac + offset
=======
    # Extract parameters
    mu_max = softplus(p[0])
    K_s = softplus(p[1])
    T_opt = p[2]
    T_width = softplus(p[3])
    pH_opt = p[4]
    pH_width = softplus(p[5])
    offset = p[6]
    b_opt = p[7]
    b_width = softplus(p[8])
    s_inh = softplus(p[9])
    
    # Normalize inputs based on known ranges
    bn = (b - 1.0) / 4.0
    tn = (temp - 30.0) / 10.0
    pn = (pH - 7.0) / 3.0
    
    # Monod equation for substrate limitation
    sfac = s / (K_s + s + 1e-9)
    
    # Haldane term for substrate inhibition (optional, controlled by s_inh)
    # This can capture non-monotonic behavior if s_inh is significant
    haldane = 1.0 / (1.0 + s_inh * s / (K_s + 1e-9))
    sfac *= haldane
    
    # Temperature bell curve
    tfac = np.exp(-0.5 * ((tn - T_opt) / (T_width + 1e-9))**2)
    
    # pH bell curve
    pfac = np.exp(-0.5 * ((pn - pH_opt) / (P_width + 1e-9))**2)
    
    # Density modulation (Gaussian bell curve)
    dfac = np.exp(-0.5 * ((bn - b_opt) / (b_width + 1e-9))**2)
    
    # Combine all factors
    return mu_max * sfac * tfac * pfac * dfac + offset
>>>>>>> REPLACE
