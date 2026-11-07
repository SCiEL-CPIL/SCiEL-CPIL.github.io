"""
Strategy: Flexible Piecewise Smooth Basis Expansion with Decoupled Temperature.

Instead of a rigid Ramberg-Osgood power law, we model the stress-strain curve
using a sum of smooth basis functions (linear + sigmoids) that can approximate
the elastic region, yield point, and hardening region flexibly.

Key Design Choices:
1. Decoupled Temperature: sigma(eps, T) = sigma_shape(eps) * scale(T) + offset(T).
   This assumes temperature primarily scales the modulus/stress level and adds
   a thermal stress offset, rather than changing the fundamental shape of the
   hardening curve. This simplifies the optimization landscape for BFGS.
2. Smooth Basis: Uses sigmoids to create "kinks" at specific strain values.
   This is more robust to variations in hardening behavior than a single power
   law and avoids NaN issues with negative strains and fractional powers.
3. Parameterization: 
   p[0]: Elastic slope
   p[1]: Yield jump amplitude
   p[2]: Yield strain center
   p[3]: Yield sigmoid sharpness (log-scale for stability)
   p[4]: Hardening 1 amplitude
   p[5]: Hardening 1 strain center
   p[6]: Hardening 2 amplitude
   p[7]: Hardening 2 strain center
   p[8]: Temp scale coefficient (linear)
   p[9]: Temp offset coefficient (linear)
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    p = params
    
    # Extract parameters
    # p[0]: Elastic slope (E_eff)
    E = p[0]
    
    # p[1]: Yield jump amplitude (Delta_sigma at yield)
    A_yield = p[1]
    
    # p[2]: Yield strain center (eps_y)
    # We constrain this to be small but positive/negative? 
    # Strain range is [-0.0019, 0.96]. Yield is likely around 0.001-0.01.
    # Let's allow it to be anywhere, but sigmoid handles it.
    eps_y = p[2]
    
    # p[3]: Yield sharpness. Use exp to ensure positive sharpness.
    # sharpness = exp(p[3]). If p[3]=0, sharpness=1. If p[3]=5, sharpness=148.
    sharpness = np.exp(p[3])
    
    # p[4]: Hardening 1 amplitude
    A_h1 = p[4]
    # p[5]: Hardening 1 center
    eps_h1 = p[5]
    
    # p[6]: Hardening 2 amplitude
    A_h2 = p[6]
    # p[7]: Hardening 2 center
    eps_h2 = p[7]
    
    # Temperature effects
    # Scale: multiplies the entire strain-based stress. 
    # Offset: adds to the stress.
    # Linear in T, centered at 0.5 for numerical stability.
    T_centered = T - 0.5
    temp_scale = 1.0 + p[8] * T_centered
    temp_offset = p[9] * T_centered
    
    # Compute basis functions
    # 1. Linear elastic term
    term_linear = E * eps
    
    # 2. Yield transition (sigmoid)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # We want a step that rises from 0 to 1 as eps crosses eps_y.
    # x = sharpness * (eps - eps_y)
    x_yield = sharpness * (eps - eps_y)
    # Clip x to avoid overflow in exp
    x_yield = np.clip(x_yield, -50, 50)
    sig_yield = 1.0 / (1.0 + np.exp(-x_yield))
    term_yield = A_yield * sig_yield
    
    # 3. Hardening 1 (sigmoid)
    # Use a fixed sharpness or derived? Let's use a fixed moderate sharpness 
    # or reuse yield sharpness scaled? 
    # To keep params < 10, let's use a fixed sharpness for hardening terms, 
    # or link it to yield sharpness. 
    # Let's use a fixed sharpness of 20.0 for hardening kinks to keep them distinct.
    sharpness_h = 20.0
    x_h1 = sharpness_h * (eps - eps_h1)
    x_h1 = np.clip(x_h1, -50, 50)
    sig_h1 = 1.0 / (1.0 + np.exp(-x_h1))
    term_h1 = A_h1 * sig_h1
    
    # 4. Hardening 2 (sigmoid)
    x_h2 = sharpness_h * (eps - eps_h2)
    x_h2 = np.clip(x_h2, -50, 50)
    sig_h2 = 1.0 / (1.0 + np.exp(-x_h2))
    term_h2 = A_h2 * sig_h2
    
    # Sum the strain-based stress components
    sigma_shape = term_linear + term_yield + term_h1 + term_h2
    
    # Apply temperature scaling and offset
    sigma = temp_scale * sigma_shape + temp_offset
    
    # Ensure finite output
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
