"""E. coli growth rate model with Monod kinetics, Gaussian T/pH responses, and biomass inhibition."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Model structure:
      mu = A * monod(s) * gauss_T(temp) * gauss_pH(pH) * logistic_inhibit(b) + c*(T-T0)*(pH-pH0) + offset
    
    Parameters (10):
      p[0]: A (amplitude, unconstrained)
      p[1]: log(Ks) -> Ks = exp(p[1])
      p[2]: T0 (optimal temperature)
      p[3]: log(sigma_T) -> sigma_T = exp(p[3])
      p[4]: pH0 (optimal pH)
      p[5]: log(sigma_pH) -> sigma_pH = exp(p[5])
      p[6]: k (inhibition sharpness)
      p[7]: b0 (inhibition midpoint)
      p[8]: c (T-pH interaction coefficient)
      p[9]: offset
    """
    A = params[0]
    Ks = np.exp(params[1])
    T0 = params[2]
    sigma_T = np.exp(params[3])
    pH0 = params[4]
    sigma_pH = np.exp(params[5])
    k = params[6]
    b0 = params[7]
    c = params[8]
    offset = params[9]

    # Monod kinetics for substrate: s / (Ks + s)
    monod = s / (Ks + s)

    # Gaussian temperature response
    gauss_T = np.exp(-0.5 * ((temp - T0) / sigma_T) ** 2)

    # Gaussian pH response
    gauss_pH = np.exp(-0.5 * ((pH - pH0) / sigma_pH) ** 2)

    # Logistic inhibition for biomass: 1 / (1 + exp(k * (b - b0)))
    # Use softplus-like stability: clip the exponent to avoid overflow
    exponent = k * (b - b0)
    exponent = np.clip(exponent, -500, 500)
    inhibit = 1.0 / (1.0 + np.exp(exponent))

    # T-pH interaction term
    interaction = c * (temp - T0) * (pH - pH0)

    # Combine
    mu = A * monod * gauss_T * gauss_pH * inhibit + interaction + offset

    return mu
