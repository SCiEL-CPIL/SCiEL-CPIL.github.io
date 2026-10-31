"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use sufficient range for Gaussian
    x = np.linspace(-15, 15, 2000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint g(0) = 0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Precompute the Gaussian weight
    gauss = np.exp(-0.5 * x**2)
    
    # Precompute H_n * gauss for each degree
    H0g = H0 * gauss
    H4g = H4 * gauss
    H8g = H8 * gauss
    H12g = H12 * gauss
    
    def objective(params):
        """
        params: [m0, m4, m8, a0, a4, a8]
        where c_i = exp(m_i) * tanh(a_i)
        This gives continuous, differentiable parameters.
        """
        try:
            m0, m4, m8, a0, a4, a8 = params
            
            # Continuous sign using tanh
            c0 = np.exp(m0) * np.tanh(a0)
            c4 = np.exp(m4) * np.tanh(a4)
            c8 = np.exp(m8) * np.tanh(a8)
            
            # Enforce g(0) = 0 => c12 is determined
            # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
            g = c0 * H0g + c4 * H4g + c8 * H8g + c12 * H12g
            
            # Compute ||g||_2^2 = ∫ g^2 dx
            norm2_sq = np.sum(g**2) * dx
            if norm2_sq < 1e-15:
                return 1e10
            
            # Compute ||g||_4^4 = ∫ |g|^4 dx
            norm4_val = np.sum(np.abs(g)**4) * dx
            if norm4_val < 1e-15:
                return 1e10
            
            # For the Fourier transform, use the property:
            # F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n * sqrt(2*pi) * H_n(ξ) * e^{-ξ^2/2}
            # So if g(x) = sum c_n H_n(x) e^{-x^2/2}, then
            # ĝ(ξ) = sqrt(2*pi) * sum c_n * (-i)^n * H_n(ξ) * e^{-ξ^2/2}
            
            # Since we only have even n (0,4,8,12), (-i)^n = 1 for all these n
            # So ĝ(ξ) = sqrt(2*pi) * (c0*H0(ξ) + c4*H4(ξ) + c8*H8(ξ) + c12*H12(ξ)) * e^{-ξ^2/2}
            
            # The L4 norm of ĝ is the same as L4 norm of the polynomial combination times sqrt(2*pi)
            # ||ĝ||_4^4 = (2*pi)^4 * ∫ |c0*H0 + c4*H4 + c8*H8 + c12*H12|^4 * e^{-2ξ^2} dξ
            # = (2*pi)^4 * ∫ |g(ξ)|^4 dξ  (since g(ξ) has the same form)
            # Wait, that's not quite right. Let me reconsider.
            
            # g(x) = P(x) * e^{-x^2/2} where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
            # ĝ(ξ) = sqrt(2*pi) * P(ξ) * e^{-ξ^2/2} = sqrt(2*pi) * g(ξ)
            
            # So ||ĝ||_4 = (2*pi)^{1/4} * ||g||_4
            # And ||ĝ||_2 = (2*pi)^{1/2} * ||g||_2
            
            # Therefore:
            # ratio = (||g||_4 * ||ĝ||_4) / (||g||_2 * ||ĝ||_2)
            #       = (||g||_4 * (2*pi)^{1/4} * ||g||_4) / (||g||_2 * (2*pi)^{1/2} * ||g||_2)
            #       = (2*pi)^{1/4} * ||g||_4^2 / (||g||_2^2)
            #       = (2*pi)^{1/4} * ||g||_4^2 / ||g||_2^2
            
            norm4_val = norm4_val ** 0.25  # ||g||_4
            norm2_val = np.sqrt(norm2_sq)  # ||g||_2
            
            ratio = (2 * np.pi) ** 0.25 * norm4_val**2 / norm2_val**2
            
            return ratio
        except Exception:
            return 1e10
    
    # Initial guess from seed values
    c0_seed = -1.13
    c4_seed = 0.04
    c8_seed = 1.0 / 3240.0
    
    m0_init = np.log(np.abs(c0_seed))
    m4_init = np.log(np.abs(c4_seed))
    m8_init = np.log(np.abs(c8_seed))
    
    # tanh(a) = sign, so a = atanh(sign) but sign is ±1, atanh(±1) is infinite
    # Use a large value instead
    a0_init = 5.0 if c0_seed < 0 else -5.0  # tanh(5) ≈ -1, tanh(-5) ≈ 1... wait
    # tanh(5) ≈ 0.9999, tanh(-5) ≈ -0.9999
    # So for negative c0, we want tanh(a0) < 0, so a0 < 0
    a0_init = -5.0 if c0_seed < 0 else 5.0
    a4_init = 5.0 if c4_seed > 0 else -5.0
    a8_init = 5.0 if c8_seed > 0 else -5.0
    
    x0 = np.array([m0_init, m4_init, m8_init, a0_init, a4_init, a8_init])
    
    # Run optimization
    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    if result.success or result.fun < objective(x0):
        m0, m4, m8, a0, a4, a8 = result.x
        c0 = np.exp(m0) * np.tanh(a0)
        c4 = np.exp(m4) * np.tanh(a4)
        c8 = np.exp(m8) * np.tanh(a8)
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([c0_seed, c4_seed, c8_seed])

# EVOLVE-BLOCK-END
