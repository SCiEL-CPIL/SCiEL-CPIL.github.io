# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Smooth-hinge Ramberg-Osgood with quadratic temperature dependence.
    sigma = E(T)*strain + H(T) * softplus(|strain| - eps_y(T), delta)^p + C0 + C1*T
    
    params:
    [0]: E0 (elastic modulus at ref T)
    [1]: E1 (linear T coefficient for E)
    [2]: E2 (quadratic T coefficient for E)
    [3]: H0 (hardening at ref T)
    [4]: H1 (linear T coefficient for H)
    [5]: ey0 (yield strain at ref T)
    [6]: ey1 (linear T coefficient for yield strain)
    [7]: delta (smoothness of yield transition)
    [8]: p (hardening exponent)
    [9]: C0 (constant offset)
    """
    E0, E1, E2 = params[0], params[1], params[2]
    H0, H1 = params[3], params[4]
    ey0, ey1 = params[5], params[6]
    delta = params[7]
    p = params[8]
    C0 = params[9]
    
    # Ensure delta is positive and sufficiently large for numerical stability
    # If delta is too small, softplus becomes a hard step with huge gradients
    # If delta is negative, log1p(exp(x/d)) can be unstable
    delta_safe = np.clip(delta, 1e-6, 100.0)
    
    # Temperature-dependent Young's modulus (quadratic in T)
    # This allows fitting non-linear thermal expansion/modulus changes
    E = E0 + E1 * temp + E2 * temp * temp
    
    # Temperature-dependent hardening modulus
    H = H0 + H1 * temp
    
    # Temperature-dependent yield strain
    eps_y = ey0 + ey1 * temp
    
    # Absolute strain for symmetric hardening
    # This handles both tension and compression without sign issues in power laws
    abs_strain = np.abs(strain)
    
    # Smooth hinge: softplus(x, delta) = delta * log1p(exp(x/delta))
    # x = abs_strain - eps_y
    # For x >> delta: ~ x
    # For x << -delta: ~ 0
    x = abs_strain - eps_y
    
    # Numerical stability for softplus
    # Clip x/delta to avoid overflow in exp()
    # exp(50) is ~ 5e21, which is safe in float64 but large
    # exp(-50) is ~ 2e-22, which underflows to 0, making log1p(0)=0, correct for softplus
    arg = x / delta_safe
    arg_clipped = np.clip(arg, -50.0, 50.0)
    
    # Calculate softplus
    # Using log1p for accuracy when arg is small
    soft_hinge = delta_safe * np.log1p(np.exp(arg_clipped))
    
    # Ensure non-negative base for power
    hinge_safe = np.maximum(soft_hinge, 1e-12)
    
    # Hardening term
    # p is the hardening exponent. Typically 0 < p < 1 for metals (sub-linear)
    # We allow p to be fitted, but constrain it slightly to avoid negative powers of 0
    # BFGS will find the optimal p.
    hardening = H * np.power(hinge_safe, p)
    
    # Elastic term
    elastic = E * strain
    
    # Combine
    # Note: C0 is the only constant offset. We don't have a separate C1*T term 
    # because E(T) and H(T) already capture linear T effects in the main terms.
    # If needed, the constant part of E(T)*strain and H(T)*hardening covers T-dependent offsets.
    stress = elastic + hardening + C0
    
    return stress
# EVOLVE-BLOCK-END
