"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 145.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible nonlinear oscillator model using 10 parameters to encode:
    - 2 sinusoidal forcing terms (amplitude, frequency, phase)
    - 2 nonlinear restoring/damping terms (amplitude, exponent)
    
    The evaluator's BFGS optimizer will fit these 10 parameters to minimize MSE.
    This approach allows the model to adapt its functional form (frequencies, 
    nonlinearity order) to the specific data, unlike a fixed linear basis.
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

    # Unpack parameters
    # p0, p1: Amplitude, Frequency for sin term 1 (driving force?)
    # p2: Phase for sin term 1
    # p3, p4: Amplitude, Frequency for sin term 2 (driving force?)
    # p5: Phase for sin term 2
    # p6, p7: Amplitude, Exponent for nonlinear restoring force (e.g., x^k)
    # p8, p9: Amplitude, Exponent for nonlinear damping force (e.g., v^k)

    # Extract with safe defaults if optimization diverges to weird values
    # We use np.clip to keep frequencies and exponents in a reasonable range
    # to avoid numerical instability in sin() or power().
    
    # Frequencies: clip to [-10, 10] to avoid high-frequency noise
    f1 = np.clip(p[1], -10.0, 10.0)
    f2 = np.clip(p[4], -10.0, 10.0)
    
    # Phases: clip to [-pi, pi]
    phi1 = np.clip(p[2], -np.pi, np.pi)
    phi2 = np.clip(p[5], -np.pi, np.pi)
    
    # Amplitudes: allow larger range
    a1 = p[0]
    a2 = p[3]
    
    # Nonlinear exponents: clip to [0.5, 5.0] to ensure stability and cover common cases (0.5=sqrt, 1=linear, 2=quadratic, 3=cubic)
    # We use abs(x)^k * sign(x) to handle negative bases with non-integer exponents
    k1 = np.clip(p[7], 0.5, 5.0)
    k2 = np.clip(p[9], 0.5, 5.0)
    
    # Nonlinear amplitudes
    c1 = p[6]
    c2 = p[8]

    # Compute terms
    # Sinusoidal terms
    term1 = a1 * np.sin(f1 * t + phi1)
    term2 = a2 * np.sin(f2 * t + phi2)
    
    # Nonlinear restoring force (x-dependent)
    # Use np.sign(x) * np.abs(x)**k to handle negative x with fractional k
    term3 = c1 * np.sign(x) * np.power(np.abs(x), k1)
    
    # Nonlinear damping force (v-dependent)
    term4 = c2 * np.sign(v) * np.power(np.abs(v), k2)
    
    # Sum them up
    out = term1 + term2 + term3 + term4
    
    # Guarantee finiteness and correct shape (one prediction per row)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
