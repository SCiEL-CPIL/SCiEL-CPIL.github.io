import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # 10s safety margin before 150s kill

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p = params
    
    # Structural Change: Separable bounded nonlinear model
    # Instead of an unbounded polynomial, use sigmoidal (logistic) and 
    # Gaussian-shaped responses that naturally bound growth rates.
    #
    # Model: growth = A * f_b(b) * f_s(s) * f_t(temp) * f_ph(pH)
    #
    # Each factor is a smooth, bounded function:
    #   f_b(b) = 1 / (1 + exp(-p[0] * (b - p[1])))   [logistic in biomass]
    #   f_s(s) = 1 / (1 + exp(-p[2] * (s - p[3])))   [logistic in substrate]
    #   f_t(temp) = exp(-((temp - p[4])^2) / (2 * p[5]^2))  [Gaussian in temp]
    #   f_ph(pH) = exp(-((pH - p[6])^2) / (2 * p[7]^2))     [Gaussian in pH]
    #
    # p[8] = amplitude (max growth rate)
    # p[9] = baseline offset
    #
    # Parameters:
    #   p[0]: steepness of biomass response
    #   p[1]: midpoint of biomass response
    #   p[2]: steepness of substrate response
    #   p[3]: midpoint of substrate response
    #   p[4]: optimal temperature
    #   p[5]: temperature width (must be > 0, use abs)
    #   p[6]: optimal pH
    #   p[7]: pH width (must be > 0, use abs)
    #   p[8]: amplitude
    #   p[9]: baseline
    
    # Compute logistic factors (clipped to avoid overflow)
    zb = p[0] * (b - p[1])
    zs = p[2] * (s - p[3])
    
    # Clip to [-50, 50] to prevent exp overflow
    zb = np.clip(zb, -50.0, 50.0)
    zs = np.clip(zs, -50.0, 50.0)
    
    f_b = 1.0 / (1.0 + np.exp(-zb))
    f_s = 1.0 / (1.0 + np.exp(-zs))
    
    # Compute Gaussian factors
    # Use abs for widths to ensure positivity
    t_width = np.abs(p[5]) + 1e-6
    ph_width = np.abs(p[7]) + 1e-6
    
    f_t = np.exp(-0.5 * ((temp - p[4]) / t_width) ** 2)
    f_ph = np.exp(-0.5 * ((pH - p[6]) / ph_width) ** 2)
    
    # Combine: multiplicative model with amplitude and baseline
    result = p[8] * f_b * f_s * f_t * f_ph + p[9]
    
    return result
