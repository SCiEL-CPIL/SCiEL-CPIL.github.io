import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Local Radial Basis Function (RBF) Expansion.
    Uses a sum of localized Gaussian kernels in strain to capture the
    non-linear stress-strain curve (yield point, hardening) smoothly.
    Temperature dependence is modeled as a linear scaling and offset.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    Returns:
    array of stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Normalize temperature to [0, 1] based on training range [0.06667, 1]
    T_min = 0.06667
    T_max = 1.0
    T_norm = (T - T_min) / (T_max - T_min)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    p = params
    
    # Parameters:
    # p[0]: Global scale (amplitude)
    # p[1]: Base offset (stress at zero strain, roughly)
    # p[2]: Temperature effect on offset
    # p[3]: Temperature effect on scale (thermal softening/stiffening)
    
    # RBF Centers (c1, c2, c3) and Widths (w1, w2, w3)
    # We use 3 Gaussian kernels to cover: elastic region, yield/transition, hardening
    # Initial guesses should be spread across the strain range [-0.002, 0.96]
    # p[4]: c1 (center 1) ~ 0.0 (elastic)
    # p[5]: w1 (width 1)  ~ 0.1 (narrow)
    # p[6]: c2 (center 2) ~ 0.2 (yield)
    # p[7]: w2 (width 2)  ~ 0.2 (medium)
    # p[8]: c3 (center 3) ~ 0.6 (hardening)
    # p[9]: w3 (width 3)  ~ 0.4 (wide)
    
    # Wait, I need to assign 10 params.
    # Let's map:
    # p[0]: scale
    # p[1]: offset
    # p[2]: T_offset
    # p[3]: T_scale
    
    # p[4]: c1
    # p[5]: w1
    # p[6]: c2
    # p[7]: w2
    # p[8]: c3
    # p[9]: w3
    
    # This leaves no amplitude parameters for the kernels!
    # The kernels will all have amplitude 1, scaled by p[0].
    # This might be too restrictive.
    
    # Alternative allocation:
    # p[0]: Global Scale
    # p[1]: Global Offset
    # p[2]: T-Offset
    # p[3]: T-Scale
    # 
    # Kernels:
    # Kernel 1: Center p[4], Width p[5]
    # Kernel 2: Center p[6], Width p[7]
    # Kernel 3: Center p[8], Width p[9]
    
    # If amplitudes are fixed to 1, the relative contributions are fixed.
    # This is risky. Let's try to fit the amplitudes instead of global offset?
    # No, offset is important for intercept.
    
    # Let's try a different split:
    # p[0]: c1
    # p[1]: w1
    # p[2]: c2
    # p[3]: w2
    # p[4]: c3
    # p[5]: w3
    # p[6]: a1 (amplitude 1)
    # p[7]: a2 (amplitude 2)
    # p[8]: a3 (amplitude 3)
    # p[9]: T_factor (linear temp correction to total stress)
    
    # This drops the explicit global offset and scale.
    # The sum of a_i * gaussian_i can provide offset if centers are far away? No.
    # If centers are in range, the sum goes to 0 at boundaries unless amplitudes are huge.
    # Stress at eps=0 is not necessarily 0.
    
    # Let's go back to the first allocation but allow the "Scale" to act as a shared amplitude,
    # and rely on the centers/widths to shape the curve.
    # Actually, let's use a Linear Term + 2 RBFs?
    # p[0]: Linear Slope (Elastic Modulus approx)
    # p[1]: Linear Intercept
    # p[2]: T-Effect on Slope
    # p[3]: T-Effect on Intercept
    # p[4]: c1
    # p[5]: w1
    # p[6]: a1
    # p[7]: c2
    # p[8]: w2
    # p[9]: a2
    
    # This looks more robust.
    # Linear part handles the initial linear elasticity and general trend.
    # RBFs handle the non-linear yield and hardening.
    
    slope = p[0] + p[2] * T_norm
    intercept = p[1] + p[3] * T_norm
    
    linear_part = slope * eps + intercept
    
    # RBF 1
    # Ensure width is positive to avoid NaN in exp
    w1 = np.abs(p[5]) + 1e-6
    rbf1 = p[6] * np.exp(-0.5 * ((eps - p[4]) / w1) ** 2)
    
    # RBF 2
    w2 = np.abs(p[8]) + 1e-6
    rbf2 = p[9] * np.exp(-0.5 * ((eps - p[7]) / w2) ** 2)
    
    stress = linear_part + rbf1 + rbf2
    
    # Safety check
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
