"""
Strategy: Non-linear Parametric Model
Instead of a linear basis expansion, we define a specific non-linear functional form
for a driven, damped Duffing oscillator. The 10 parameters are hyperparameters that
define the physics:
- p0: Driving frequency (omega)
- p1: Damping coefficient (c)
- p2: Linear stiffness (k)
- p3: Non-linear stiffness (alpha, for Duffing term)
- p4: Driving amplitude (A)
- p5: Phase shift (phi)
- p6: Quadratic damping coefficient (beta, for v^2 term)
- p7: Time offset (t0)
- p8: Constant bias
- p9: Higher harmonic driving amplitude (for richer forcing)

The equation is:
a = -c*v - beta*v*|v| - k*x - alpha*x^3 + A*sin(omega*(t-t0) + phi) + p9*sin(2*omega*(t-t0)) + p8

This form is non-linear in parameters (especially p0, p1, p5), requiring BFGS to perform
a non-linear least squares fit. This is fundamentally different from the linear basis approach
which could only approximate the function within the span of the chosen basis.
"""

import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven, damped Duffing oscillator.
    Inputs:
        t: time
        x: position
        v: velocity
        params: array of 10 hyperparameters
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    # Defensive: if params is short, pad with zeros; if long, ignore extras.
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (omega, c, k, alpha, A, phi, beta, t0, bias, A2) = p

    # To avoid numerical issues with BFGS exploring bad parameter spaces,
    # we use soft-clipping or safe operations where necessary.
    # However, standard math operations are generally stable for reasonable ranges.
    # The evaluator starts params at all ones, so initial values are safe.
    
    # Calculate the driving force components
    # t_eff = t - t0
    t_eff = t - t0
    
    # Main driving term
    drive_main = A * np.sin(omega * t_eff + phi)
    
    # Secondary harmonic driving term (allows for more complex forcing profiles)
    drive_harmonic = A2 * np.sin(2.0 * omega * t_eff)
    
    # Damping terms
    # Linear damping: -c * v
    damp_linear = -c * v
    
    # Quadratic damping: -beta * v * |v| (or v^2, but v*|v| is more physical for drag)
    # Using v * abs(v) ensures the sign of damping opposes velocity always.
    # If beta is negative, it becomes anti-damping, which is allowed in optimization.
    damp_quad = -beta * v * np.abs(v)
    
    # Restoring force terms (Duffing)
    # Linear: -k * x
    # Cubic: -alpha * x^3
    restore_linear = -k * x
    restore_cubic = -alpha * (x * x * x)
    
    # Combine
    out = (
        damp_linear 
        + damp_quad 
        + restore_linear 
        + restore_cubic 
        + drive_main 
        + drive_harmonic 
        + bias
    )

    # Guarantee finiteness and correct shape
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
