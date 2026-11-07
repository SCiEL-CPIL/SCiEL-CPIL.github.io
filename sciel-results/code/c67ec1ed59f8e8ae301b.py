"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model for a driven Duffing/Van der Pol oscillator.
    
    The acceleration (target) is modeled as:
    a = -k1*x - k2*x^3 + k3*v + k4*x*v + F0*sin(w*t + phi) + c
    
    This uses 10 parameters:
    p0: c (constant bias)
    p1: k1 (linear stiffness)
    p2: k2 (cubic stiffness)
    p3: k3 (linear damping)
    p4: k4 (nonlinear damping coefficient)
    p5: F0 (forcing amplitude)
    p6: w (forcing frequency)
    p7: phi (forcing phase)
    p8: scaling factor for x (to help BFGS with scale)
    p9: scaling factor for v (to help BFGS with scale)
    
    By using physical structures (sinusoidal driving, cubic nonlinearity), 
    we provide a much better inductive bias than a generic polynomial basis.
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

    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9) = p

    # Scaled variables to improve conditioning for BFGS
    # The inputs are in specific ranges, but parameters start at 1.
    # Scaling helps the optimizer navigate the landscape.
    x_scaled = x * p8
    v_scaled = v * p9

    # Physical terms
    # Linear restoring force
    term_linear_restoring = -p1 * x_scaled
    
    # Nonlinear restoring force (Duffing)
    term_nonlinear_restoring = -p2 * (x_scaled ** 3)
    
    # Linear damping
    term_linear_damping = p3 * v_scaled
    
    # Nonlinear damping (Van der Pol style)
    term_nonlinear_damping = p4 * x_scaled * v_scaled
    
    # Driving force (Sinusoidal)
    # Note: t is in [30, 50]. The frequency p6 needs to be able to capture the driving frequency.
    # If the true frequency is known, this would be fixed. Here it is a parameter.
    # We assume the driving force is F0 * sin(w*t + phi)
    term_driving = p5 * np.sin(p6 * t + p7)
    
    # Constant bias
    term_bias = p0

    # Combine
    out = (
        term_bias 
        + term_linear_restoring 
        + term_nonlinear_restoring 
        + term_linear_damping 
        + term_nonlinear_damping 
        + term_driving
    )

    # Guarantee finiteness and correct shape (one prediction per row)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
