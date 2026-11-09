import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Structural change: multiplicative mechanistic model
    # rate = mu_max * f_substrate(s) * f_temp(temp) * f_pH(pH) * f_biomass(b)
    #
    # Parameters (10 total):
    # p[0]: mu_max (max growth rate, use softplus to keep positive)
    # p[1]: Ks (half-saturation constant for substrate, softplus)
    # p[2]: T_opt (optimal temperature)
    # p[3]: T_width (temperature response width, softplus)
    # p[4]: pH_opt (optimal pH)
    # p[5]: pH_width (pH response width, softplus)
    # p[6]: b_scale (scale for biomass effect)
    # p[7]: b_power (exponent for biomass effect)
    # p[8]: offset (additive offset)
    # p[9]: s_inh (substrate inhibition coefficient, softplus)
    
    # Helper: stable softplus
    def softplus(x):
        x_clipped = np.clip(x, -50, 50)
        return np.log1p(np.exp(x_clipped))
    
    # Extract parameters with appropriate constraints
    mu_max = softplus(p[0])
    Ks = softplus(p[1])
    T_opt = p[2]
    T_width = softplus(p[3])
    pH_opt = p[4]
    pH_width = softplus(p[5])
    b_scale = p[6]
    b_power = p[7]
    offset = p[8]
    s_inh = softplus(p[9])
    
    # Substrate effect: Monod with inhibition (Haldane)
    # f_s = s / (Ks + s) * 1 / (1 + s/Ks)  -- actually standard Haldane: s/(Ks+s+s^2/Ks)
    # Simpler: Monod s/(Ks+s), plus inhibition term
    s_term = s / (Ks + s)
    inh_term = 1.0 / (1.0 + s / s_inh)
    f_substrate = s_term * inh_term
    
    # Temperature effect: Gaussian around T_opt
    f_temp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_width ** 2))
    
    # pH effect: Gaussian around pH_opt
    f_pH = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_width ** 2))
    
    # Biomass effect: power law, normalized
    b_norm = np.clip(b / 5.0, 0.01, 1.0)  # normalize b to [0.01, 1]
    f_biomass = np.power(b_norm, b_power)
    
    # Combine multiplicatively
    result = mu_max * f_substrate * f_temp * f_pH * f_biomass + offset
    
    return result
