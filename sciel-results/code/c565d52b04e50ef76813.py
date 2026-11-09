import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad
import time

def _compute_uncertainty_value(a0, a4, a8):
    """
    Computes the uncertainty product A(g)*A(F(g)) for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    subject to g(0) = 0.
    """
    # H_n(0) values for even n:
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 5040
    # H12(0) = 103950
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 5040.0
    h12_0 = 103950.0
    
    # Constraint g(0) = 0:
    # a0*h0_0 + a4*h4_0 + a8*h8_0 + a12*h12_0 = 0
    # a12 = -(a0 + 12*a4 + 5040*a8) / 103950
    a12 = -(a0 + h4_0*a4 + h8_0*a8) / h12_0
    
    # Define g(x)
    def g(x):
        return (a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x) + 
                a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)) * np.exp(-x**2 / 2.0)
    
    # Define g'(x) analytically to avoid numerical differentiation errors
    # d/dx [H_n(x) exp(-x^2/2)] = 2n H_{n-1}(x) exp(-x^2/2)
    # So g'(x) = (2*0*a0*H_{-1} + 2*4*a4*H_3 + 2*8*a8*H_7 + 2*12*a12*H_11) * exp(-x^2/2)
    # Note: H_{-1} is 0.
    def gp(x):
        return (2*4*a4 * eval_hermite(3, x) + 
                2*8*a8 * eval_hermite(7, x) + 
                2*12*a12 * eval_hermite(11, x)) * np.exp(-x**2 / 2.0)

    # Integrate |g|^2, x^2|g|^2, |g'|^2
    # Using quad for high precision
    try:
        I_g2, _ = quad(lambda x: g(x)**2, -np.inf, np.inf, limit=100)
        I_x2g2, _ = quad(lambda x: x**2 * g(x)**2, -np.inf, np.inf, limit=100)
        I_gp2, _ = quad(lambda x: gp(x)**2, -np.inf, np.inf, limit=100)
    except Exception:
        return 1e10

    if I_g2 < 1e-15:
        return 1e10

    # Delta_x^2 = <x^2> - <x>^2. By symmetry <x>=0.
    delta_x2 = I_x2g2 / I_g2
    
    # Delta_k^2 = <k^2> - <k>^2. By symmetry <k>=0.
    # Using Parseval and properties of Fourier transform:
    # int k^2 |F(k)|^2 dk = int |g'(x)|^2 dx (for F(k) = 1/sqrt(2pi) int g(x) e^{-ikx} dx)
    # Wait, let's check the constant.
    # If F(k) = (1/sqrt(2pi)) int g(x) e^{-ikx} dx, then int |F(k)|^2 dk = int |g(x)|^2 dx.
    # And int k^2 |F(k)|^2 dk = int |g'(x)|^2 dx.
    # So Delta_k^2 = I_gp2 / I_g2.
    delta_k2 = I_gp2 / I_g2
    
    if delta_x2 < 0 or delta_k2 < 0:
        return 1e10
        
    return np.sqrt(delta_x2 * delta_k2)

def construct():
    start_time = time.time()
    deadline = start_time + 150.0  # Safety margin
    
    # Baseline coefficients (State of the art)
    baseline = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Compute baseline value
    base_val = _compute_uncertainty_value(*baseline)
    best_coeffs = baseline.copy()
    best_val = base_val
    
    # Attempt a very brief local optimization to see if we can improve
    # We use Nelder-Mead with strict limits
    try:
        if time.time() < deadline - 30.0:
            def objective(params):
                if time.time() > deadline - 10.0:
                    return best_val
                return _compute_uncertainty_value(*params)
            
            # Start from baseline
            # Small step size to ensure we don't wander too far
            options = {
                'maxiter': 100,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            res = minimize(objective, baseline, method='Nelder-Mead', options=options)
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                
    except Exception:
        pass
    
    # If optimization didn't help or failed, return baseline
    return best_coeffs
