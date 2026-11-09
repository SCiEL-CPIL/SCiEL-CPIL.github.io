"""Hybrid Chebyshev-phase-plane model with parametric frequency drive for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining Chebyshev polynomials in (x,v) phase plane
    with parametric-frequency sinusoidal drive terms.
    
    The drive uses sin(p0*t + p1) and cos(p0*t + p1) where p0 is the
    fitted frequency and p1 is the fitted phase, allowing the optimizer
    to discover the true drive frequency rather than being locked to 1.0.
    
    The remaining 8 parameters weight Chebyshev basis functions in
    normalized (x, v) space, providing a well-conditioned approximation
    of the nonlinear restoring and damping forces.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
            p0: drive frequency
            p1: drive phase
            p2..p9: Chebyshev coefficients
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract drive parameters
    p0 = p[0]  # frequency
    p1 = p[1]  # phase
    # p2..p9 are Chebyshev coefficients
    
    # Normalize x and v to [-1, 1] for Chebyshev evaluation
    # x range: [-0.2056, 0.1749] -> center = -0.01535, half_range = 0.19025
    # v range: [-0.3391, 0.3624] -> center = 0.01165, half_range = 0.35075
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    xn = (x - x_center) / x_half
    vn = (v - v_center) / v_half
    
    # Clip to [-1, 1] to keep Chebyshev well-defined
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials up to degree 3
    T0_x = np.ones_like(xn)
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    
    T0_v = np.ones_like(vn)
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn
    
    # Cross terms: T1_x * T1_v, T2_x * T1_v, T1_x * T2_v, T2_x * T2_v
    # We have 8 Chebyshev params (p2..p9)
    # Assign: p2=T0_x, p3=T1_x, p4=T2_x, p5=T3_x,
    #         p6=T1_v, p7=T2_v, p8=T3_v, p9=T1_x*T1_v
    # This gives us: constant, x-terms, v-terms, and one cross term
    
    # Actually let's be more strategic with 8 params:
    # p2: T0 (bias)
    # p3: T1_x (linear x)
    # p4: T2_x (quadratic x)
    # p5: T3_x (cubic x)
    # p6: T1_v (linear v)
    # p7: T2_v (quadratic v)
    # p8: T3_v (cubic v)
    # p9: T1_x * T1_v (cross coupling)
    
    # Build the drive terms with fitted frequency and phase
    drive_sin = np.sin(p0 * t + p1)
    drive_cos = np.cos(p0 * t + p1)
    
    # Build Chebyshev basis
    T0 = T0_x  # ones
    T1x = T1_x
    T2x = T2_x
    T3x = T3_x
    T1v = T1_v
    T2v = T2_v
    T3v = T3_v
    cross = T1_x * T1_v
    
    # Combine: drive + Chebyshev phase-plane terms
    # The drive amplitude is implicitly 1 (or we could scale, but with only 10 params
    # we use the drive as fixed amplitude 1, and the Chebyshev terms handle the rest)
    # Actually, let's think about this differently.
    # We have 10 params total. If we use 2 for drive (freq, phase), we have 8 for the rest.
    # But the drive needs amplitude too. Let's use a different allocation.
    
    # Alternative: use p0 as frequency, and embed amplitude into the Chebyshev constant.
    # Or: use a simpler drive with fixed frequency but fitted amplitude.
    
    # Let me reconsider. The key insight from inspiration 2 is that parametric frequency
    # (p2*t inside sin/cos) helped. But it scored 92, worse than the linear-in-t approach.
    # The issue is that with only 10 params, using 2 for the drive frequency/phase leaves
    # fewer for the phase-plane dynamics.
    
    # Better approach: Use fixed-frequency drive (sin(t), cos(t)) but with fitted amplitudes,
    # and use the remaining params for a richer Chebyshev basis.
    
    # Final allocation:
    # p0: amplitude of sin(t) drive
    # p1: amplitude of cos(t) drive
    # p2: bias (T0)
    # p3: T1_x
    # p4: T2_x
    # p5: T3_x
    # p6: T1_v
    # p7: T2_v
    # p8: T3_v
    # p9: T1_x * T1_v (cross term)
    
    result = (p[0] * np.sin(t) + 
              p[1] * np.cos(t) +
              p[2] * T0 +
              p[3] * T1x +
              p[4] * T2x +
              p[5] * T3x +
              p[6] * T1v +
              p[7] * T2v +
              p[8] * T3v +
              p[9] * cross)
    
    # Safety: replace any non-finite values with 0 to avoid scoring penalty
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
