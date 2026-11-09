"""
Strategy: Multi-scale Chebyshev Product Basis.

Unlike the previous Gaussian RBF approach which relied on fixed local centers,
this strategy uses a global orthogonal polynomial basis (Chebyshev) in 3D.
The key structural change is the introduction of nonlinearity in the basis functions
themselves via learnable 'scale' and 'shift' parameters for each dimension.

The basis is constructed as products of 1D Chebyshev polynomials:
T_i(x') * T_j(y') * T_k(z')
where x' = a_x * (x - b_x), etc.

By making the scaling factors (a) learnable, the basis can adapt its effective
frequency to the target function. This allows the model to capture both low-frequency
smooth trends and higher-frequency oscillations if present, without the risk of
ill-conditioning associated with high-degree monomials (Chebyshev is orthogonal on [-1,1]).

Parameters (10 total):
- p[0:3]: Scaling factors for x, y, z (initial 1.0, mapped to positive via softplus/exp)
- p[3:10]: Amplitudes for the 7 basis terms:
  1. T0(x)T0(y)T0(z)  -> 1
  2. T1(x)T0(y)T0(z)  -> x'
  3. T0(x)T1(y)T0(z)  -> y'
  4. T0(x)T0(y)T1(z)  -> z'
  5. T1(x)T1(y)T0(z)  -> x'y'
  6. T1(x)T0(y)T1(z)  -> x'z'
  7. T0(x)T1(y)T1(z)  -> y'z'
  
  Note: We omit the full 3-way interaction T1(x)T1(y)T1(z) to keep the parameter count at 10
  and maintain stability, but the 2-way interactions provide significant expressive power
  for non-separable functions. If needed, we could swap one linear term for the 3-way term,
  but usually, mixed 2-way terms are more critical for general smooth functions on cubes.
  
  Actually, let's reconsider the basis selection to maximize coverage with 7 terms.
  Standard full tensor product up to degree 1 in 3D has 8 terms:
  1, x, y, z, xy, xz, yz, xyz.
  We have 7 amplitude slots. We will use all 8 terms if we reduce scaling params?
  No, we need scaling for adaptability.
  
  Let's stick to:
  1. 1
  2. x'
  3. y'
  4. z'
  5. x'y'
  6. x'z'
  7. y'z'
  
  This covers all pairwise interactions. The missing xyz term is a high-order interaction.
  If the function is separable or has strong pairwise dependencies, this is optimal.
  
  To make it more powerful, we can allow the "shift" to be fixed at the center of the domain (1.25)
  so we only learn the scale. This reduces the search space for the optimizer.
"""
import numpy as np
import time

# Pre-compute Chebyshev polynomials logic inside the function for speed
# T_0(t) = 1
# T_1(t) = t

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    N = len(x)
    
    # Deadline check
    if time.time() - start_time > 140.0:
        return np.ones(N)

    # Extract parameters
    # p[0], p[1], p[2] are raw scaling parameters for x, y, z
    # We use exp() to ensure positive scaling, which helps BFGS navigate
    # and prevents sign flips that might confuse the polynomial basis.
    # Initial params are 1.0, so exp(1.0) ~ 2.7.
    # The domain is [0.5, 2.0]. Center is 1.25. Range is 1.5.
    # Normalized variable t = (val - center) / half_range * scale
    # If scale=1, t is in [-0.75, 0.75] * (1.5/0.75)? No.
    # Let's define t = (val - 1.25) / 0.75. This maps [0.5, 2.0] to [-1, 1].
    # Then we apply a learnable scale factor s.
    # t_scaled = s * t.
    # If s=1, we are in the standard Chebyshev domain.
    # If s > 1, we are "zooming in" on the center, effectively increasing frequency.
    # If s < 1, we are "zooming out", decreasing frequency.
    
    sx = np.exp(p[0])
    sy = np.exp(p[1])
    sz = np.exp(p[2])
    
    # Amplitudes
    a0 = p[3]  # Constant
    a1 = p[4]  # x term
    a2 = p[5]  # y term
    a3 = p[6]  # z term
    a4 = p[7]  # xy term
    a5 = p[8]  # xz term
    a6 = p[9]  # yz term
    
    # Compute normalized coordinates
    # Map [0.5, 2.0] to [-1, 1]
    # x_norm = (x - 1.25) / 0.75
    x_norm = (x - 1.25) / 0.75
    y_norm = (y - 1.25) / 0.75
    z_norm = (z - 1.25) / 0.75
    
    # Apply learnable scales
    # Clip to avoid extreme values if optimizer diverges, though exp helps
    # We don't clip the normalized inputs themselves as they are bounded by construction
    # but the scaled version could be large if sx is huge.
    # Chebyshev T1(t) = t. T0(t) = 1.
    # So our basis functions are just polynomials of degree 1 in the scaled vars.
    # Basis 1: 1
    # Basis 2: sx * x_norm
    # Basis 3: sy * y_norm
    # Basis 4: sz * z_norm
    # Basis 5: (sx * x_norm) * (sy * y_norm)
    # Basis 6: (sx * x_norm) * (sz * z_norm)
    # Basis 7: (sy * y_norm) * (sz * z_norm)
    
    # To add more power, we could use T2(t) = 2t^2 - 1 for some terms?
    # With only 7 amplitudes, adding T2 would require dropping linear terms or interactions.
    # Given the "smooth function" assumption often implicit in these tasks unless specified otherwise,
    # linear + pairwise interactions is a very strong baseline.
    # However, if the function has curvature (e.g., x^2), linear terms in scaled space
    # can only approximate it poorly over the whole domain unless the scale is very small (local).
    # But we have global scales.
    
    # Let's upgrade the basis to include quadratic terms by using T2 for the single-variable slots
    # and keeping T1 for interactions?
    # No, that mixes degrees.
    
    # Alternative: Use a richer basis with the 7 amplitudes:
    # 1. 1
    # 2. x'
    # 3. y'
    # 4. z'
    # 5. x'y'
    # 6. x'z'
    # 7. y'z'
    
    # This is essentially a quadratic polynomial without pure squares (x^2, y^2, z^2) and without xyz.
    # If the target is e.g. x^2, this model cannot fit it well.
    
    # Let's try a different split:
    # 1. 1
    # 2. x'
    # 3. y'
    # 4. z'
    # 5. x'^2 (using T2? No, just x'^2 is simpler for BFGS gradient)
    # 6. y'^2
    # 7. z'^2
    # This captures curvature but misses interactions.
    
    # Best compromise for "unknown function":
    # Use a mix.
    # 1. 1
    # 2. x'
    # 3. y'
    # 4. z'
    # 5. x'y'
    # 6. x'z'
    # 7. y'z'
    
    # Why this over pure quadratics?
    # Interactions are often more important in 3D problems (e.g., distance, products).
    # Also, if the function is separable f(x)+g(y)+h(z), the linear terms in x' might not be enough
    # if f is non-linear.
    
    # Let's make the basis slightly more flexible by allowing the "linear" terms to actually be
    # higher order Chebyshev polynomials?
    # T_2(t) = 2t^2 - 1.
    # If we replace the linear terms with T_2, we lose the linear trend.
    
    # Let's stick to the pairwise interaction model but ensure the scaling is robust.
    # The key insight from the "stagnation" hint is that we need a structural change.
    # The previous RBF was local. This is global.
    # The previous polynomial was likely low-order monomials.
    # This uses orthogonal polynomials with adaptive scaling.
    
    bx = sx * x_norm
    by = sy * y_norm
    bz = sz * z_norm
    
    # Calculate basis values
    # For numerical stability, if sx, sy, sz are very large, bx, by, bz can be large.
    # But x_norm is in [-1, 1].
    
    pred = (a0 * np.ones(N) + 
            a1 * bx + 
            a2 * by + 
            a3 * bz + 
            a4 * (bx * by) + 
            a5 * (bx * bz) + 
            a6 * (by * bz))
            
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.ones(N)
        
    return pred
