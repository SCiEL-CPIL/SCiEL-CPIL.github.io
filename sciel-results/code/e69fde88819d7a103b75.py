"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    Omega_DE0 = 1.0 - Om0
    ratio_DE = (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(3.0 * wa * (a - 1.0))
    Om_a = Om0 * a**(-3.0) / (Om0 * a**(-3.0) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    def integrand(x):
        a_x = np.exp(x)
        return _f_of_a(a_x, coeffs, Om0, w0, wa)
    integral_val, _ = quad(integrand, ln_a, 0.0, limit=200)
    return np.exp(-integral_val)

def _solve_growth_reference(a_eval, Om0, w0, wa, n_steps=2000):
    """Solve the growth ODE numerically for reference."""
    a_arr = np.linspace(0.001, 1.0, n_steps)
    
    # Precompute Om(a) and H^2 for all points
    Om_arr = np.zeros_like(a_arr)
    H2_arr = np.zeros_like(a_arr)
    for i, ai in enumerate(a_arr):
        Om_arr[i] = _Om_of_a(ai, Om0, w0, wa)
        H2_arr[i] = Om_arr[i] * ai**(-3.0)
    
    # Compute q(a) = 0.5 * d ln H2 / d ln a
    q_arr = np.zeros_like(a_arr)
    for i in range(1, n_steps - 1):
        da = a_arr[i + 1] - a_arr[i - 1]
        dH2 = H2_arr[i + 1] - H2_arr[i - 1]
        q_arr[i] = 0.5 * a_arr[i] * dH2 / (da * H2_arr[i])
    q_arr[0] = q_arr[1]
    q_arr[-1] = q_arr[-2]
    
    # Solve ODE: D' = v, v' = -(3/a - q(a))*v + (3/2)*Om(a)*D
    D = np.zeros_like(a_arr)
    v = np.zeros_like(a_arr)
    
    # Initial conditions at early times (matter domination)
    D[0] = a_arr[0]
    v[0] = 1.0
    
    dt = a_arr[1] - a_arr[0]
    for i in range(n_steps - 1):
        ai = a_arr[i]
        Di = D[i]
        vi = v[i]
        qi = q_arr[i]
        Omi = Om_arr[i]
        
        def rhs(D_val, v_val):
            D_prime = v_val
            v_prime = -(3.0 / ai - qi) * v_val + (1.5 * Omi) * D_val
            return D_prime, v_prime
        
        k1_D, k1_v = rhs(Di, vi)
        k2_D, k2_v = rhs(Di + 0.5 * dt * k1_D, vi + 0.5 * dt * k1_v)
        k3_D, k3_v = rhs(Di + 0.5 * dt * k2_D, vi + 0.5 * dt * k2_v)
        k4_D, k4_v = rhs(Di + dt * k3_D, vi + dt * k3_v)
        
        D[i + 1] = Di + (dt / 6.0) * (k1_D + 2 * k2_D + 2 * k3_D + k4_D)
        v[i + 1] = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    
    # Normalize so D(1) = 1
    D_norm = D / D[-1]
    v_norm = v / D[-1]
    
    # Interpolate to find D(a_eval) and f(a_eval) = v/D
    idx = np.searchsorted(a_arr, a_eval)
    if idx <= 0:
        return D_norm[0], v_norm[0] / D_norm[0]
    if idx >= n_steps:
        return D_norm[-1], v_norm[-1] / D_norm[-1]
    
    t = (a_eval - a_arr[idx - 1]) / (a_arr[idx] - a_arr[idx - 1])
    D_val = D_norm[idx - 1] + t * (D_norm[idx] - D_norm[idx - 1])
    v_val = v_norm[idx - 1] + t * (v_norm[idx] - v_norm[idx - 1])
    f_val = v_val / D_val
    return D_val, f_val

def _reference_f(a, Om0, w0, wa):
    """Reference f and D from numerical GR solution."""
    D_ref, f_ref = _solve_growth_reference(a, Om0, w0, wa)
    return f_ref, D_ref

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline for safety margin
    
    # Test cosmologies (w0, wa, Om0) - broader range for better generalization
    test_models = [
        (-1.0, 0.0, 0.3),      # LCDM
        (-1.0, 0.1, 0.3),      # CPL with wa=0.1
        (-1.0, 0.2, 0.3),      # CPL with wa=0.2
        (-1.0, 0.3, 0.3),      # CPL with wa=0.3
        (-1.0, 0.5, 0.3),      # CPL with wa=0.5
        (-0.9, 0.0, 0.3),      # CPL with w0=-0.9
        (-0.8, 0.0, 0.3),      # CPL with w0=-0.8
        (-1.1, 0.0, 0.3),      # CPL with w0=-1.1
        (-1.2, 0.0, 0.3),      # CPL with w0=-1.2
        (-0.9, 0.2, 0.3),      # Mixed CPL
        (-0.8, -0.2, 0.3),     # Negative wa
        (-1.2, 0.5, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.27),     # LCDM with different Om
        (-1.0, 0.0, 0.32),     # LCDM with higher Om
        (-0.95, 0.1, 0.28),    # Near-LCDM CPL
        (-1.05, -0.1, 0.32),   # Another near-LCDM
        (-1.0, 0.15, 0.29),    # Intermediate
        (-0.85, 0.25, 0.31),   # Another mixed
    ]
    
    # a values to check - denser sampling
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # Precompute references for all test models and a values
    print("Computing references...")
    ref_cache = {}
    for (w0, wa, Om0) in test_models:
        for a in a_values:
            key = (w0, wa, Om0, a)
            f_ref, D_ref = _reference_f(a, Om0, w0, wa)
            ref_cache[key] = (f_ref, D_ref)
    
    print(f"References computed in {time.time() - start_time:.1f}s")
    
    def objective(coeffs):
        """Max relative error over all test models and a values."""
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            for a in a_values:
                key = (w0, wa, Om0, a)
                f_ref, D_ref = ref_cache[key]
                
                # Compute our approximation
                f_approx = _f_of_a(a, coeffs, Om0, w0, wa)
                D_approx = _D_normalized(a, coeffs, Om0, w0, wa)
                
                # Relative errors
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                else:
                    err_f = abs(f_approx - f_ref)
                
                if abs(D_ref) > 1e-10:
                    err_D = abs(D_approx - D_ref) / abs(D_ref)
                else:
                    err_D = abs(D_approx - D_ref)
                
                max_err = max(max_err, err_f, err_D)
        
        return max_err
    
    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_err = objective(x0)
    print(f"Initial error: {best_err:.6f}")
    
    # Try Nelder-Mead optimization with multiple starting points
    starting_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.55, 0.05, 0.02, 0.05, 0.02, 0.05],
        [0.52, 0.08, 0.03, 0.08, 0.03, 0.08],
        [0.58, -0.05, 0.04, -0.03, 0.04, -0.02],
    ]
    
    for i, sp in enumerate(starting_points):
        if time.time() > deadline - 10.0:
            break
        sp_arr = np.array(sp)
        try:
            res = minimize(
                objective,
                sp_arr,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if res.fun < best_err:
                best_coeffs = res.x
                best_err = res.fun
                print(f"  Start {i}: improved to {best_err:.6f}")
        except Exception as e:
            print(f"  Start {i}: error {e}")
    
    # Final refinement with tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_err:
                best_coeffs = res.x
                best_err = res.fun
                print(f"  Final refinement: {best_err:.6f}")
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    print(f"Final error: {best_err:.6f}, coeffs: {best_coeffs}")
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
