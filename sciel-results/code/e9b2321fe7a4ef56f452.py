import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0

    p = params

    # Structural change: Multiplicative biological model
    # Parameters are transformed to map initial guess (all ones) to biologically plausible values
    
    # p[0]: Max growth rate
    mu_max = p[0]
    
    # p[1]: K_s (Monod constant). Start 1 -> exp(1-2) = 0.37
    K_s = np.exp(p[1] - 2.0)
    
    # p[2]: T_opt. Start 1 -> sigmoid(1 + 1.2) ~ 0.9 (37C)
    T_opt = 1.0 / (1.0 + np.exp(-(p[2] + 1.2)))
    
    # p[3]: Sigma_t. Start 1 -> exp(1-2) = 0.37
    sigma_t = np.exp(p[3] - 2.0)
    
    # p[4]: pH_opt. Start 1 -> sigmoid(1 - 1) = 0.5 (pH 7)
    pH_opt = 1.0 / (1.0 + np.exp(-(p[4] - 1.0)))
    
    # p[5]: Sigma_pH. Start 1 -> exp(1-2) = 0.37
    sigma_ph = np.exp(p[5] - 2.0)
    
    # p[6]: Biomass inhibition coeff. Start 1 -> 0.37
    b_inh = np.exp(p[6] - 2.0)
    
    # p[7]: Substrate inhibition coeff. Start 1 -> 0.37
    s_inh = np.exp(p[7] - 2.0)
    
    # p[8]: Temp shift due to biomass. Start 1 -> 0.1
    shift_t = p[8] * 0.1
    
    # p[9]: pH shift due to biomass. Start 1 -> 0.1
    shift_ph = p[9] * 0.1

    # Calculate components
    # Monod + Substrate Inhibition
    s_term = s_norm / (K_s + s_norm + s_inh * s_norm * s_norm + 1e-8)
    
    # Temperature Bell Shape with biomass-dependent shift
    T_center = T_opt - shift_t * b_norm
    t_term = np.exp(-((t_norm - T_center) ** 2) / (2.0 * sigma_t * sigma_t + 1e-8))
    
    # pH Bell Shape with biomass-dependent shift
    pH_center = pH_opt - shift_ph * b_norm
    ph_term = np.exp(-((ph_norm - pH_center) ** 2) / (2.0 * sigma_ph * sigma_ph + 1e-8))
    
    # Biomass Inhibition
    b_term = 1.0 / (1.0 + b_inh * b_norm * b_norm)
    
    # Combine
    y = mu_max * s_term * t_term * ph_term * b_term
    
    return y


