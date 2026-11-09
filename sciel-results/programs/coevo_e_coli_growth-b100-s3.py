import numpy as np
import time

def equation(b, s, temp, pH, params):
    """
    Closed-form expression for E. coli growth rate.
    Inputs:
        b: biomass/density (range ~1-5)
        s: substrate concentration (range ~1-40)
        temp: temperature (range ~1-40)
        pH: pH level (range ~2-12)
    Params:
        p[0]: T_center (normalized 0-1)
        p[1]: T_width (log-space, exp for scale)
        p[2]: pH_center (normalized 0-1)
        p[3]: pH_width (log-space, exp for scale)
        p[4]: K_s (log-space, exp for scale)
        p[5]: mu_max (log-space, exp for scale)
        p[6]: d1 (linear coeff for b)
        p[7]: d2 (quad coeff for b)
        p[8]: d3 (cubic coeff for b)
        p[9]: offset
    """
    p = params
    
    # Unpack params
    T_c = p[0]
    T_w = np.exp(p[1])  # Positive width
    pH_c = p[2]
    pH_w = np.exp(p[3])  # Positive width
    K_s = np.exp(p[4])   # Positive half-saturation constant
    mu_max = np.exp(p[5]) # Positive max rate
    d1 = p[6]
    d2 = p[7]
    d3 = p[8]
    offset = p[9]
    
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # T: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Monod substrate
    sfac = s / (K_s + s + 1e-9)
    
    # Temperature bell (Gaussian on normalized scale)
    tfac = np.exp(-0.5 * ((xT - T_c) / T_w)**2)
    
    # pH bell (Gaussian on normalized scale)
    pfac = np.exp(-0.5 * ((xPH - pH_c) / pH_w)**2)
    
    # Density modulation (Cubic)
    dfac = 1.0 + d1 * xb + d2 * xb**2 + d3 * xb**3
    
    # Final prediction
    return mu_max * sfac * tfac * pfac * dfac + offset
