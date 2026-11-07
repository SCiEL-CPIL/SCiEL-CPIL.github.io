import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack params
    Vmax = params[0]
    Ks = np.abs(params[1]) + 1e-6
    T_opt = params[2]
    T_width = np.abs(params[3]) + 0.1
    pH_opt = params[4]
    pH_width = np.abs(params[5]) + 0.05
    K_b = np.abs(params[6]) + 1e-6
    alpha_b = params[7]
    gamma = params[8]
    offset = params[9]

    # Substrate: Monod
    # s is in [1, 39.98]
    S_term = s / (Ks + s)
    
    # Temperature: Gaussian
    # temp is in [1, 4.999] ? Wait, the prompt says inputs are [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12].
    # The order is b, s, temp, pH.
    # b: [1, 4.998]
    # s: [1, 4.999] ?? No, the prompt lists inputs in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12].
    # Let's re-read carefully: "Inputs in training range, in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]."
    # This corresponds to b, s, temp, pH.
    # So b is ~1-5.
    # s is ~1-5.
    # temp is ~1-40. (This makes sense for E. coli, 1-40 C)
    # pH is ~2-12. (This makes sense for pH)
    
    # Okay, so temp range is wide (1-40). Gaussian is good.
    # pH range is wide (2-12). Gaussian is good.
    # s range is narrow (1-5). Monod is good.
    # b range is narrow (1-5). Saturation/Inhibition is good.

    T_term = np.exp(-((temp - T_opt) / T_width) ** 2)
    
    pH_term = np.exp(-((pH - pH_opt) / pH_width) ** 2)
    
    # Density: Saturation with exponent
    # b is in [1, 5]
    B_term = (b / (K_b + b)) ** alpha_b
    
    # Combine
    growth = Vmax * S_term * T_term * pH_term * B_term
    
    return gamma * growth + offset
