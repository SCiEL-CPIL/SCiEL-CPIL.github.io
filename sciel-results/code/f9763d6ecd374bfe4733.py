"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # Piecewise-linear approximation using ramp functions (clipped linear terms)
    # We use 5 ramp breakpoints in strain space and 2 in temperature space
    # Total parameters: 5 (strain ramps) + 2 (temp ramps) + 3 (base linear/offset) = 10
    
    # Strain ramps: max(0, strain - c_i)
    # Temp ramps: max(0, temp - d_i)
    # Base: a0 + a1*strain + a2*temp
    
    # Extract parameters
    # params[0:3] -> base coefficients (offset, strain slope, temp slope)
    # params[3:5] -> strain ramp breakpoints (c1, c2)
    # params[5:7] -> strain ramp slopes (s1, s2)
    # params[7:9] -> temp ramp breakpoints (d1, d2)
    # params[9]   -> temp ramp slope (t1)
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    
    c1 = params[3]
    c2 = params[4]
    s1 = params[5]
    s2 = params[6]
    
    d1 = params[7]
    d2 = params[8]
    t1 = params[9]
    
    # Compute ramp functions
    # For numerical stability and to help BFGS, we ensure smoothness by using
    # softplus-like approximations? No, the directive says clipped linear terms.
    # But strict clipping (max(0, x)) is not differentiable at 0.
    # BFGS can handle this with subgradients or by avoiding the kink, but it's risky.
    # Let's use a smooth approximation of the ramp: x * sigmoid(k*x) / k ?
    # Or simply max(0, x) is often fine for BFGS if initialized well, but the kink
    # can cause issues. A better differentiable ramp is:
    # ramp(x) = (x + sqrt(x^2 + eps)) / 2
    # This is smooth and approximates max(0, x) well for small eps.
    
    eps = 1e-6
    
    def smooth_ramp(x):
        return (x + np.sqrt(x**2 + eps)) / 2.0
    
    # Strain ramps
    r1 = smooth_ramp(strain - c1)
    r2 = smooth_ramp(strain - c2)
    
    # Temp ramps
    t_r1 = smooth_ramp(temp - d1)
    t_r2 = smooth_ramp(temp - d2)
    
    # Combine
    # We only have 2 strain ramp slopes and 2 temp ramp breakpoints but only 1 temp ramp slope param left.
    # Let's re-allocate:
    # params[0:3] -> base
    # params[3:5] -> strain breakpoints
    # params[5:7] -> strain slopes
    # params[7:9] -> temp breakpoints
    # params[9]   -> temp slope (applied to both? or just one?)
    # Actually, let's use the last param for a combined temp effect or just one temp ramp.
    # To keep it simple and within 10 params:
    # Let's do 3 strain ramps and 1 temp ramp?
    # 3 strain ramps need 3 breakpoints and 3 slopes = 6 params.
    # 1 temp ramp needs 1 breakpoint and 1 slope = 2 params.
    # Base = 2 params (slope, offset).
    # Total = 6 + 2 + 2 = 10. This fits perfectly.
    
    # Re-extract for the new plan:
    # params[0] -> offset
    # params[1] -> base strain slope
    # params[2:5] -> strain ramp breakpoints (c1, c2, c3)
    # params[5:8] -> strain ramp slopes (s1, s2, s3)
    # params[8]   -> temp ramp breakpoint (d1)
    # params[9]   -> temp ramp slope (t1)
    
    a0 = params[0]
    a1 = params[1]
    
    c1, c2, c3 = params[2], params[3], params[4]
    s1, s2, s3 = params[5], params[6], params[7]
    
    d1 = params[8]
    t1 = params[9]
    
    # Compute smooth ramps
    r1 = smooth_ramp(strain - c1)
    r2 = smooth_ramp(strain - c2)
    r3 = smooth_ramp(strain - c3)
    
    t_r1 = smooth_ramp(temp - d1)
    
    # Prediction
    pred = a0 + a1 * strain + s1 * r1 + s2 * r2 + s3 * r3 + t1 * t_r1
    
    return pred
# EVOLVE-BLOCK-END
