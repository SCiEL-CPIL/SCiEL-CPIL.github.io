import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] roughly to help optimization stability
    # Ranges: b[1,5], s[1,5], temp[1,40], pH[2,12]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    # Monod term for Substrate
    # p[0] is Ks (normalized). Ensure positive Ks via softplus or square?
    # BFGS starts at 1. If p[0] becomes negative, Monod breaks.
    # Use exp(p[0]) to ensure positive Ks.
    Ks = np.exp(p[0])
    monod_s = s_n / (Ks + s_n)
    
    # Temperature term: Gaussian
    # p[1] is width parameter (inverse variance), p[2] is center
    # If p[1] is large, it's a narrow peak. If small, flat.
    # Start at 1.
    f_temp = np.exp(-p[1] * (t_n - p[2])**2)
    
    # pH term: Gaussian
    # p[3] is width parameter, p[4] is center
    f_ph = np.exp(-p[3] * (ph_n - p[4])**2)
    
    # Density term: Inhibition
    # p[5] is inhibition strength
    # Use 1 / (1 + p[5] * b_n) to ensure positive and bounded
    f_b = 1.0 / (1.0 + np.exp(p[5]) * b_n) # exp(p[5]) ensures positive coefficient
    
    # Max growth rate
    mu_max = np.exp(p[6]) # Ensure positive
    
    # Baseline / Death rate
    baseline = p[7]
    
    # Interaction term? Maybe substrate and temp interact?
    # Let's add a simple interaction between s and T to use more params if needed, 
    # or just leave it. We have 8 params used.
    # Let's add a quadratic correction for temperature?
    # Or a linear interaction between T and pH?
    
    # Let's try to use p[8] and p[9] for fine-tuning the shape.
    # Perhaps a power law on the Monod term?
    # monod_s_powered = monod_s ** p[8]
    
    # Let's keep it simple first. If we have spare params, we can add a linear term for b?
    # No, f_b covers it.
    
    # Let's add a term that allows the peak height to be adjusted independently of the width?
    # Already have mu_max.
    
    # What if the pH optimum shifts with temperature?
    # f_ph = exp(-p[3] * (ph_n - (p[4] + p[8]*(t_n - 0.5)))**2)
    
    # Let's try a simpler additive model with interactions to avoid multiplicative zero-crossing issues during optimization.
    # y = p0 * monod * f_temp * f_ph * f_b + p7
    
    prediction = mu_max * monod_s * f_temp * f_ph * f_b + baseline
    
    # Ensure finite
    return prediction
