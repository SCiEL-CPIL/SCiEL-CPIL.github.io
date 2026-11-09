"""E. coli growth rate model: hybrid quadratic response surface with biological priors.

Combines the smooth quadratic response surface (inspiration 1, score 1.1513) with
biologically-motivated structure from the Monod model (current, score 0.8647).

Strategy:
- Use normalized inputs for numerical stability with BFGS.
- Include linear, quadratic, and key interaction terms.
- Add a Monod-style substrate saturation term as a smooth nonlinear feature.
- Include a Gaussian temperature optimum as a smooth nonlinear feature.
- This gives BFGS a smooth landscape while retaining biological structure.

Parameter allocation (10 params):
  p[0]: intercept
  p[1]: linear b coefficient
  p[2]: linear s coefficient
  p[3]: linear temp coefficient
  p[4]: linear pH coefficient
  p[5]: quadratic temp coefficient
  p[6]: quadratic pH coefficient
  p[7]: Monod Ks (half-saturation)
  p[8]: Monod amplitude (multiplies the saturation term)
  p[9]: temp-pH interaction coefficient
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Hybrid model: quadratic response surface + Monod substrate saturation + Gaussian temp.
    
    Inputs:
        b: population density, range [1, 4.998]
        s: substrate concentration, range [1, 4.999]
        temp: temperature, range [1.002, 39.98]
        pH: pH, range [2, 12]
    
    Parameters:
        params: 10 fitted parameters
    """
    p = params
    
    # Normalize inputs for numerical stability
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    
    # Monod substrate saturation term (smooth, bounded [0, 1])
    Ks = np.abs(p[7]) + 1e-6
    monod = (s * 0.5) / (Ks + s * 0.5)  # scale s by 0.5 to keep in reasonable range
    
    # Gaussian temperature factor (smooth, bounded [0, 1])
    T_center = 37.0  # typical E. coli optimal temperature
    T_width = 10.0
    temp_gauss = np.exp(-((temp - T_center) ** 2) / (2 * T_width ** 2))
    
    # Quadratic terms
    T2 = T_n * T_n
    pH2 = pH_n * pH_n
    
    # Interaction term
    T_pH = T_n * pH_n
    
    # Combine: linear + quadratic + biological features
    mu = (p[0]
          + p[1] * b_n
          + p[2] * s_n
          + p[3] * T_n
          + p[4] * pH_n
          + p[5] * T2
          + p[6] * pH2
          + p[8] * monod * temp_gauss
          + p[9] * T_pH)
    
    return mu
# EVOLVE-BLOCK-END
