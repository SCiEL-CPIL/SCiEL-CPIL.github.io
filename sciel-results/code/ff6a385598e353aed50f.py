"""
Strategy: Rational Function (Padé-like) Ansatz.
Instead of a sum of basis functions (current) or a physical power-law (Ramberg-Osgood),
we model stress as a ratio of two polynomials in strain:
  sigma = N(eps, T) / D(eps, T)
where N and D are low-degree polynomials.
This form is extremely flexible for smooth curves, can model asymptotes (if D -> 0)
or plateaus, and often fits monotonic S-curves or hardening curves very well with
fewer parameters than high-order polynomials or splines. It is structurally different
from the additive basis approach.

We use:
N(eps, T) = p0 + p1*eps + p2*eps^2 + p3*eps^3
D(eps, T) = 1 + q1*eps + q2*eps^2

Temperature dependence is injected by making coefficients linear functions of T_norm.
To keep parameter count at 10 and ensure stability:
- We use a shared structure.
- p = params
- T_norm is in [0,1].

Let's define:
N = (p0 + p1*eps + p2*eps^2 + p3*eps^3) * (1 + p4*T_norm)
D = 1 + (p5 + p6*T_norm)*eps + (p7 + p8*T_norm)*eps^2

Wait, that's 9 params. We have 10.
Let's add a global scale or offset.
sigma = (p9 * N) / D

This ensures the denominator is close to 1 for small eps, avoiding division by zero
unless p5/p6/p7/p8 drive it to zero, which BFGS should avoid if initialized at 1
(leading to D = 1 + 1*eps + 1*eps^2, which is positive for eps > -0.5).
Given strain range [-0.002, 0.96], D is likely safe.

This rational form is known to approximate many physical curves (like logistic,
saturation) very compactly.
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    
    Returns:
    array of stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Normalize temperature
    T_norm = (T - 0.06667) / (1.0 - 0.06667)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    p = params
    
    # Extract parameters
    # p[0]: N constant term
    # p[1]: N linear term
    # p[2]: N quadratic term
    # p[3]: N cubic term
    # p[4]: T effect on N (global scale of numerator)
    # p[5]: D linear term (base)
    # p[6]: D linear term (T effect)
    # p[7]: D quadratic term (base)
    # p[8]: D quadratic term (T effect)
    # p[9]: Global scale for final stress
    
    # Numerator
    N = p[0] + p[1]*eps + p[2]*eps**2 + p[3]*eps**3
    N = N * (1.0 + p[4]*T_norm)
    
    # Denominator
    # Keep D close to 1 to avoid instability, but allow curvature
    D = 1.0 + (p[5] + p[6]*T_norm)*eps + (p[7] + p[8]*T_norm)*eps**2
    
    # Safety: if D is close to 0, clip it to avoid inf/nan
    # Since we don't know the sign, we clip magnitude
    D_safe = np.where(np.abs(D) < 1e-6, np.sign(D)*1e-6 + (1 - np.sign(D)*1e-6), D)
    # Simpler safety: just add a small epsilon if D is very small
    # But np.where is faster. Let's use a robust clip.
    # If D is 0, we want it to be small.
    # Let's just use: D_safe = np.where(D == 0, 1e-6, D)
    # Actually, BFGS might push D to 0.
    # Let's use: D_safe = np.where(np.abs(D) < 1e-4, np.sign(D)*1e-4, D)
    # If D is exactly 0, sign is 0, so we need to handle that.
    # D_safe = np.where(np.abs(D) < 1e-4, 1e-4 * np.where(D < 0, -1, 1), D)
    
    D_safe = np.where(np.abs(D) < 1e-4, 1e-4 * np.where(D < 0, -1.0, 1.0), D)
    
    sigma = p[9] * (N / D_safe)
    
    # Ensure finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
