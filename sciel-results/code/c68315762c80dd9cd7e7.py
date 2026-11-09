"""
Strategy: Adaptive Multi-Scale Gaussian Process (Kernel) Regression.

This strategy departs from the fixed smooth basis (Chebyshev/MLP) by using a kernel
method with a radial basis function (RBF) kernel.
Rationale:
1. Local Flexibility: Unlike global bases (polynomials) or global non-linearities (MLP),
   RBF kernels allow the model to fit local features, sharp transitions, or localized bumps
   that global smooth functions struggle with.
2. Parameter Interpretation: The 10 parameters are mapped to 5 "basis functions" (Gaussians),
   each with a center $c_i$, width $\sigma_i$, and amplitude $a_i$. Plus a global offset $b$.
   This gives 5*3 + 1 = 16 parameters, which is too many.
   We need exactly 10.
   
   Let's adjust:
   Option A: 3 Gaussians (3*3=9) + 1 offset = 10.
   Option B: 4 Gaussians with shared width (4*2 + 1 width + 1 offset = 10).
   Option C: 5 Gaussians with fixed centers? No, centers need to be fitted.
   
   Let's go with Option A: 3 Gaussians.
   Params:
   p[0], p[1], p[2]: centers c1, c2, c3
   p[3], p[4], p[5]: widths sigma1, sigma2, sigma3
   p[6], p[7], p[8]: amplitudes a1, a2, a3
   p[9]: offset b
   
   Model: y = a1 * exp(-0.5 * ((x-c1)/sigma1)^2) + a2 * exp(-0.5 * ((x-c2)/sigma2)^2) + 
          a3 * exp(-0.5 * ((x-c3)/sigma3)^2) + b
   
   Issue: BFGS starting from all ones might struggle if the targets are far from the initial
   Gaussian peaks (which are at x=1). Also, if sigma is small, gradients can vanish or explode.
   To mitigate:
   - Initialize centers spread across the domain? We can't change initialization, it's all ones.
   - However, we can reparameterize to make the landscape smoother.
   - Use a "soft" Gaussian: exp(-k * (x-c)^2) where k is the inverse variance.
   - If sigma is close to 0, the function becomes a spike. If sigma is large, it becomes flat.
   - Starting at 1: c=1, sigma=1, a=1, b=1.
   
   Alternative: RBF with fixed centers?
   If we fix centers at -2, 0, 2, we only need amplitudes and widths.
   3 centers * (1 amp + 1 width) + 1 offset = 7 params. We have 3 left.
   Maybe 4 centers? 4 * (1 amp + 1 width) + 1 offset = 9 params. 1 left for a linear term?
   
   Let's try a hybrid:
   Fixed centers at -2, 0, 2.
   Params:
   a1, a2, a3 (amplitudes)
   s1, s2, s3 (widths)
   b (offset)
   m (linear slope)
   q (quadratic coeff)
   Total: 3 + 3 + 1 + 1 + 1 = 9. One more?
   Maybe a constant term is already b.
   Let's add a 4th Gaussian at center 1? No, symmetric is better.
   
   Let's stick to the 3 Gaussian + polynomial tail idea, but ensure stability.
   Actually, the "Adaptive" part suggests the centers should move.
   But BFGS from all ones is a big risk for non-convex centers.
   
   Let's try a different structure: **Spline-like Basis with Local Support**.
   Or, simply, **Fourier Series with Higher Order Terms**?
   The previous MLP failed (3019 vs 4397). The Chebyshev+Fourier hybrid worked well (4397).
   The inspiration says mixing Chebyshev + Trig + Tanh + Gaussian got 4427.
   
   So, a **Mixture of Experts** or **Multi-Resolution Basis** is likely better.
   
   New Strategy: **Multi-Resolution Gaussian Basis with Fixed Centers**.
   Fixed centers prevent the optimizer from getting lost in the center landscape.
   We use 5 fixed centers: -2.5, -1.25, 0, 1.25, 2.5.
   Each has an amplitude $a_i$ and a width $\sigma_i$.
   5 * 2 = 10 parameters. No offset?
   If we include an offset, we need 9 amplitudes/widths.
   Let's use 4 centers: -2, -1, 1, 2.
   4 * (1 amp + 1 width) = 8 params.
   1 offset.
   1 linear term (to handle trends).
   Total: 10.
   
   Model:
   y = a1*G(x, c1, s1) + a2*G(x, c2, s2) + a3*G(x, c3, s3) + a4*G(x, c4, s4) + m*x + b
   
   Where G(x, c, s) = exp(-0.5 * ((x-c)/s)^2).
   
   This covers local bumps and global trends.
   BFGS should handle this reasonably well as the landscape is smoother than variable centers.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the output of a Multi-Resolution Gaussian Basis model.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
            [a1, s1, a2, s2, a3, s3, a4, s4, m, b]
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    
    # Extract parameters
    a1, s1 = params[0], params[1]
    a2, s2 = params[2], params[3]
    a3, s3 = params[4], params[5]
    a4, s4 = params[6], params[7]
    m = params[8]
    b = params[9]
    
    # Fixed centers
    c1, c2, c3, c4 = -2.0, -1.0, 1.0, 2.0
    
    # Ensure widths are positive to avoid numerical issues in division/squaring
    # Using np.maximum to keep it differentiable-ish for BFGS (though max is non-diff at 0, 
    # BFGS usually handles small kinks better than hard crashes, but let's use a softplus-like 
    # or just a small epsilon floor. Since BFGS uses gradients, a hard floor at 0.01 is safe 
    # as the gradient is 0 there, preventing steps into negative widths which cause complex numbers 
    # or huge values. Actually, if s is negative, s^2 is positive, so the Gaussian is valid. 
    # The only issue is if s=0. So we just need to avoid s=0.
    # However, if s is negative, the "width" is effectively |s|. 
    # Let's just square the width parameter to ensure it's positive and smooth.
    s1_sq = s1 * s1 + 1e-6
    s2_sq = s2 * s2 + 1e-6
    s3_sq = s3 * s3 + 1e-6
    s4_sq = s4 * s4 + 1e-6
    
    # Compute Gaussians
    # G(x, c, s) = exp(-0.5 * ((x-c)^2 / s^2))
    g1 = np.exp(-0.5 * ((x - c1) ** 2) / s1_sq)
    g2 = np.exp(-0.5 * ((x - c2) ** 2) / s2_sq)
    g3 = np.exp(-0.5 * ((x - c3) ** 2) / s3_sq)
    g4 = np.exp(-0.5 * ((x - c4) ** 2) / s4_sq)
    
    # Linear and offset
    linear = m * x
    offset = b
    
    # Combine
    y = a1 * g1 + a2 * g2 + a3 * g3 + a4 * g4 + linear + offset
    
    return y

# EVOLVE-BLOCK-END
