import numpy as np
import scipy.optimize as opt
import time

# Precompute Gauss-Hermite nodes and weights
# We need integrals of e^{-2 pi x^2} * polynomial
# Change variable u = sqrt(2 pi) x
# Integral = 1/sqrt(2 pi) * Integral e^{-u^2} P(u/sqrt(2 pi))^2 du
# Use hermgauss(200) for high accuracy

def get_gh_nodes(n=100):
    # Returns x, w for integral of e^{-x^2} f(x)
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w

def compute_var_product(c):
    c0, c4, c8 = c
    c12 = (c0 - 12*c4 + 168*c8) / 2880.0
    
    # Polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # H0=1, H4=16x^4-48x^2+12, H8=256x^8-1344x^6+2688x^4-1344x^2+168
    # H12=4096x^12 - 46080x^10 + 230400x^8 - 483840x^6 + 403200x^4 - 129600x^2 + 2880
    
    # We evaluate P at x = u / sqrt(2 pi)
    # Let alpha = sqrt(2 pi)
    alpha = np.sqrt(2 * np.pi)
    x_gh, w_gh = get_gh_nodes(150)
    x_vals = x_gh / alpha
    
    # Calculate H_k(x_vals)
    # Using explicit formulas for speed
    x2 = x_vals**2
    x4 = x2**2
    x6 = x4*x2
    x8 = x4**2
    x10 = x8*x2
    x12 = x6**2
    
    H0 = np.ones_like(x_vals)
    H4 = 16*x4 - 48*x2 + 12
    H8 = 256*x8 - 1344*x6 + 2688*x4 - 1344*x2 + 168
    H12 = 4096*x12 - 46080*x10 + 230400*x8 - 483840*x6 + 403200*x4 - 129600*x2 + 2880
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    
    # f(x) = e^{-pi x^2} P(x)
    # |f|^2 = e^{-2 pi x^2} P^2
    # Integral of e^{-2 pi x^2} P^2 dx = 1/alpha * Integral e^{-u^2} P(u/alpha)^2 du
    # Weighted sum: sum(w_i * P(x_i)^2)
    integrand_sq = w_gh * (P**2)
    norm_sq = np.sum(integrand_sq) / alpha
    
    if norm_sq < 1e-12:
        return 1e10
        
    # Integral x^2 |f|^2 dx
    integrand_x2 = w_gh * (x_vals**2) * (P**2)
    moment_x2 = np.sum(integrand_x2) / alpha
    
    var_x2 = moment_x2 / norm_sq
    
    # Derivative f'(x)
    # f(x) = e^{-pi x^2} P(x)
    # f'(x) = e^{-pi x^2} (P'(x) - 2 pi x P(x))
    # |f'|^2 = e^{-2 pi x^2} (P' - 2 pi x P)^2
    # Need P'(x)
    H0_p = 0
    H4_p = 64*x_vals**3 - 96*x_vals
    H8_p = 2048*x_vals**7 - 8064*x_vals**5 + 10752*x_vals**3 - 2688*x_vals
    H12_p = 49152*x_vals**11 - 460800*x_vals**9 + 1843200*x_vals**7 - 2903040*x_vals**5 + 1612800*x_vals**3 - 259200*x_vals
    
    P_p = c0*H0_p + c4*H4_p + c8*H8_p + c12*H12_p
    
    term = P_p - 2 * np.pi * x_vals * P
    integrand_dp = w_gh * (term**2)
    moment_dp = np.sum(integrand_dp) / alpha
    
    # Delta xi^2 = 1/(4 pi^2) * <(f')^2> / <f^2>
    var_xi2 = (1.0 / (4 * np.pi**2)) * (moment_dp / norm_sq)
    
    var_prod = np.sqrt(var_x2 * var_xi2)
    return var_prod

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Initial guess
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_c = c_seed
    best_val = compute_var_product(c_seed)
    
    # Multi-start
    rng = np.random.RandomState(42)
    
    # Grid of starts
    starts = [c_seed]
    # Add random perturbations
    for _ in range(20):
        c_rand = c_seed + rng.normal(0, 0.1, size=3)
        starts.append(c_rand)
        
    # L-BFGS-B
    for c0 in starts:
        if time.time() > deadline:
            break
        try:
            res = opt.minimize(compute_var_product, c0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
            
    return best_c
