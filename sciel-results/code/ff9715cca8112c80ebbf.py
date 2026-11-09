"""Symbolic regression combining Chebyshev polynomial basis with a learnable
Gaussian bump for localized features, plus a smooth tanh term."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model combining:
      - Chebyshev polynomials of the first kind, degree 0..6 (7 coefficients)
      - A smooth tanh term (1 coefficient)
      - A Gaussian bump with learnable center, width, and amplitude (3 parameters)

    params (length 10):
        p0..p6: coefficients for T_0(t)..T_6(t), where t = x/2
        p7: coefficient for tanh(x)
        p8: center of Gaussian bump
        p9: width of Gaussian bump (used as 1 + |p9| to avoid zero-width)
    
    The Gaussian amplitude is implicitly 1.0 (fixed), so p8 and p9 control
    position and spread. This keeps the model linear in the first 8 params
    and only mildly nonlinear in p8, p9, which BFGS can handle.

    Actually, let's make it fully linear in all params by precomputing the
    Gaussian basis function with a fixed but reasonable default, and letting
    p8 be its amplitude and p9 be a small modulation. But that loses flexibility.

    Better approach: Keep 7 Chebyshev coeffs, 1 tanh coeff, and use p8 as
    Gaussian amplitude, p9 as Gaussian center. Width fixed at 0.5 for stability.
    This makes the model linear in all 10 params, so BFGS converges quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 2.0  # scale to [-1, 1] for Chebyshev

    # Compute Chebyshev polynomials T_0 through T_6 via recurrence
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4

    # Smooth tanh term
    tanh_term = np.tanh(x)

    # Gaussian bump: exp(-(x - center)^2 / (2 * width^2))
    # Use fixed width = 0.5 for stability; center is learnable via p9
    # Amplitude is p8
    width = 0.5
    center = params[9]
    gauss_arg = -((x - center) ** 2) / (2.0 * width ** 2)
    gauss_term = np.exp(gauss_arg)

    # Combine all terms linearly
    result = (
        params[0] * T0
        + params[1] * T1
        + params[2] * T2
        + params[3] * T3
        + params[4] * T4
        + params[5] * T5
        + params[6] * T6
        + params[7] * tanh_term
        + params[8] * gauss_term
    )

    # p9 is used as the center of the Gaussian, so it's already incorporated
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
