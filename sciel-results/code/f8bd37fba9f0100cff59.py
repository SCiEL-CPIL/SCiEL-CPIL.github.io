import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the normalized uncertainty bound, with H12 determined by g(0)=0.
    
    Approach:
    1. Use the property that for n divisible by 4, F[H_n e^{-x^2/2}] = H_n e^{-x^2/2}.
    2. Thus g is its own Fourier transform (real, even).
    3. Objective becomes minimizing (||g||_1^2) / (||g||_2^2).
    4. Scale invariance allows fixing a0=1.
    5. Use deterministic local optimization (Nelder-Mead) on the 2D space of (a4, a8).
    """
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def get_a12(a0, a4, a8):
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    def g_func(x, a0, a4, a8):
        a12 = get_a12(a0, a4, a8)
        # g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2)
        poly = a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x) + a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)
        return poly * np.exp(-x**2 / 2)

    def objective(params):
        a4, a8 = params
        a0 = 1.0
        
        # Compute L1 norm: integral of |g(x)|
        # We use quad for high accuracy. 
        # g(x) is even, so integrate 0 to inf and double.
        def abs_g(x):
            return np.abs(g_func(x, a0, a4, a8))
        
        # Integration bounds: Gaussian decays fast. 10 is sufficient.
        l1_half, _ = quad(lambda x: abs_g(x), 0, 10, limit=200, epsabs=1e-10, epsrel=1e-10)
        l1_g = 2 * l1_half
        
        # Compute L2 norm squared: integral of g(x)^2
        def g_sq(x):
            val = g_func(x, a0, a4, a8)
            return val * val
            
        l2_sq_half, _ = quad(lambda x: g_sq(x), 0, 10, limit=200, epsabs=1e-10, epsrel=1e-10)
        l2_sq_g = 2 * l2_sq_half
        
        if l2_sq_g < 1e-12:
            return 1e10
            
        return (l1_g * l1_g) / l2_sq_g

    # Initial guess: 
    # We want g(x) to be positive to maximize the ratio? No, we want to minimize L1/L2.
    # Actually, for a fixed L2, spreading out reduces L1. Concentrating increases L1.
    # Wait, the uncertainty principle usually bounds the product of spreads.
    # The "Uncertainty Inequality" task likely refers to minimizing the constant in 
    # ||f||_1 ||F||_1 >= C ||f||_2^2.
    # We want to find the function that achieves the lowest bound, i.e., minimizes the ratio.
    # Gaussian alone: H0. a4=a8=0.
    # Let's start near the Gaussian but slightly perturbed.
    initial_guess = np.array([0.0, 0.0])
    
    # Use Nelder-Mead for robustness in 2D without gradients
    try:
        res = minimize(objective, initial_guess, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success or res.fun < 1e10:
            a4_opt, a8_opt = res.x
        else:
            # Fallback
            a4_opt, a8_opt = 0.0, 0.0
            
    except Exception:
        a4_opt, a8_opt = 0.0, 0.0
        
    a0_opt = 1.0
    return np.array([a0_opt, a4_opt, a8_opt])
