import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model for E. coli growth rate.
    
    Structural Changes:
    1. Input Normalization: Scales inputs to ~[0, 1] to improve BFGS conditioning.
    2. Monod Kinetics for Substrate: Uses s/(Ks+s) form for biological saturation.
       Ks is parameterized as exp(p2 - 2) to ensure positivity and reasonable init.
    3. Quadratic Terms for Temperature, pH, and Density: Retains flexibility 
       from the best-performing quadratic model while benefiting from normalization.
    4. Interaction: Includes T*pH interaction.
    
    params layout (10 constants):
      p[0]: Bias
      p[1]: Monod scale (max growth contribution from substrate)
      p[2]: Log-Ks parameter for Monod (Ks = exp(p2 - 2))
      p[3]: Linear coeff for normalized Temperature
      p[4]: Quadratic coeff for normalized Temperature
      p[5]: Linear coeff for normalized pH
      p[6]: Quadratic coeff for normalized pH
      p[7]: Linear coeff for normalized Density (b)
      p[8]: Quadratic coeff for normalized Density (b)
      p[9]: Interaction coeff for normalized T * pH
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to approximately [0, 1]
    # b: [1, 5] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 5] -> [0, 1]
    s_norm = (s - 1.0) / 4.0
    # temp: [1, 40] -> [0, 1]
    T_norm = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    pH_norm = (pH - 2.0) / 10.0
    
    # Monod kinetics for substrate
    # p2 is initialized to 1.0. exp(1.0 - 2.0) = exp(-1) ~ 0.37.
    # This is a reasonable Ks for a normalized substrate in [0, 1].
    Ks = np.exp(p2 - 2.0)
    monod_s = s_norm / (Ks + s_norm)
    
    # Construct prediction
    y = p0
    y += p1 * monod_s
    y += p3 * T_norm + p4 * (T_norm ** 2)
    y += p5 * pH_norm + p6 * (pH_norm ** 2)
    y += p7 * b_norm + p8 * (b_norm ** 2)
    y += p9 * (T_norm * pH_norm)
    
    return y
