import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs for stability
    # b: [1, 5] -> [-1, 1]
    b_norm = (b - 3.0) / 2.0
    
    # s: [1, 40] -> use raw for Monod, p1 is Ks
    # temp: [1, 50] -> centered at 30, scaled by 10
    # Optimal ~37 -> (37-30)/10 = 0.7
    t_norm = (temp - 30.0) / 10.0
    
    # pH: [4, 10] -> centered at 7, scaled by 2
    # Optimal ~7 -> (7-7)/2 = 0
    ph_norm = (pH - 7.0) / 2.0
    
    # Monod term
    # Avoid division by zero if p1 is negative and s is small? 
    # p1 starts at 1. BFGS might make it negative.
    # Monod is undefined for Ks < 0 physically, but mathematically s/(p1+s) can work if p1+s != 0.
    # To be safe, we can use np.maximum(p1, 1e-6) or just let it be.
    # Given BFGS, it might explore negative Ks.
    # Let's stick to raw p1 but ensure no nan.
    denom = p1 + s
    # If denom is close to 0, monod blows up.
    # We can add a small epsilon.
    monod = s / (denom + 1e-6)
    
    # Temperature Gaussian Peak
    # p2 is the center in normalized units.
    # p3 is the width parameter (inverse variance).
    # Clip exponent to prevent underflow/overflow issues, though exp(-large) -> 0 is fine.
    temp_exponent = -p3 * (t_norm - p2)**2
    temp_term = np.exp(temp_exponent)
    
    # pH Gaussian Peak
    ph_exponent = -p5 * (ph_norm - p4)**2
    ph_term = np.exp(ph_exponent)
    
    # Biomass effect: Quadratic
    b_term = 1.0 + p6 * b_norm + p7 * b_norm**2
    
    # Linear corrections in log-space for extra flexibility
    # This allows for skew or shifts not captured by pure Gaussian
    log_corr = p8 * t_norm + p9 * ph_norm
    corr_term = np.exp(log_corr)
    
    return p0 * monod * temp_term * ph_term * b_term * corr_term
