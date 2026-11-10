"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    # H(a)/H0 = sqrt(Om0*a^-3 + Omega_DE(a))
    # Omega_DE(a) = Omega_DE0 * a^(-3(1+w0+wa)) where w_eff = w0+wa*(1-a)
    # Actually for CPL: rho_DE(a) = rho_DE0 * a^(-3(1+w0+wa*(1-a)))
    # This is tricky - let's use the standard form
    # For CPL: p = w0 + wa*(1-a), so rho_DE ∝ a^(-3(1+w0+wa*(1-a)))
    # This doesn't have a simple power-law form. Let's integrate numerically or use approximation.
    # Actually, the standard CPL has rho_DE(a) = rho_DE0 * exp(-3*wa*(1-a)) * a^(-3(1+w0))
    # But for the growth equation, we need Om(a) = Om0*a^-3 / (Om0*a^-3 + Omega_DE(a))
    
    # For simplicity and to match the task's definition where u=1+w0+wa*(1-a),
    # I'll use the effective w approach: treat as if w_eff = w0+wa*(1-a)
    # Then rho_DE(a) ∝ a^(-3(1+w_eff)) but this is approximate
    
    # Let me use the exact CPL: w(a) = w0 + wa*(1-a)
    # rho_DE(a)/rho_DE(1) = exp(-3*integral_1^a (1+w(a'))/a' da')
    # = exp(-3*integral_1^a (1+w0+wa*(1-a'))/a' da')
    # = exp(-3*integral_1^a (1+w0)/a' da' - 3*wa*integral_1^a (1-a')/a' da')
    # = exp(-3(1+w0)*ln(a) - 3*wa*integral_1^a (1/a' - 1) da')
    # = exp(-3(1+w0)*ln(a) - 3*wa*(ln(a) - (a-1)))
    # = exp(-3(1+w0)*ln(a) - 3*wa*ln(a) + 3*wa*(a-1))
    # = a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    
    Omega_DE0 = 1.0 - Om0
    ratio_DE = (a ** (-3*(1+w0+wa))) * np.exp(3*wa*(a-1))
    Om_a = Om0 * a**(-3) / (Om0 * a**(-3) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    # Clamp to valid range
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
    # Integrate f(exp(x)) from ln_a to 0
    def integrand(x):
        a_x = np.exp(x)
        return _f_of_a(a_x, coeffs, Om0, w0, wa)
    integral_val, _ = quad(integrand, ln_a, 0.0)
    return np.exp(-integral_val)

def _reference_f(a, Om0, w0, wa):
    """Reference f from numerical GR solution (approximate using known fits)."""
    # For the reference, I'll use a high-accuracy approximation
    # The growth rate f = d ln D / d ln a satisfies f' + f^2 = (3/2)Om(a)
    # This is a Riccati equation. We can solve it numerically.
    # But for speed, let's use a well-known approximation or solve ODE
    # Actually, let me solve the ODE numerically for accuracy
    
    # Solve f' + f^2 = (3/2)Om(a) with boundary condition f(1) ≈ 0.55*Om(1)^0.55
    # This is tricky. Let me use the known fact that for flat LCDM, 
    # f ≈ 0.55*Om^0.55 is quite good. For CPL, we need to adjust.
    
    # I'll use a numerical integration approach for the reference
    # The growth equation: D'' + (3/a - q(a))D' - (3/2)Om(a)D = 0
    # where q(a) = (1/2)(d ln H^2/d ln a)
    
    # For simplicity and to have a "reference", I'll use the constant gamma 
    # approximation as a baseline, but the actual evaluator uses numerical GR.
    # Since I don't have the exact GR reference, I'll optimize against 
    # a set of test cosmologies using a more accurate method.
    
    # Let me implement a simple ODE solver for the reference
    def _solve_growth_ODE(a_eval, Om0, w0, wa, n_steps=1000):
        """Solve the growth ODE numerically."""
        # D'' + (3/a - q(a))D' - (3/2)Om(a)D = 0
        # Let D' = v, then D'' = v'
        # v' + (3/a - q(a))v - (3/2)Om(a)D = 0
        # D' = v
        
        # q(a) = (1/2) d ln H^2 / d ln a
        # H^2 = H0^2 (Om0*a^-3 + Omega_DE(a))
        # d ln H^2 / d ln a = (a/H^2) dH^2/da
        # For CPL, this is complex. Let me use numerical differentiation.
        
        a_arr = np.linspace(0.01, 1.0, n_steps)
        H2 = np.zeros_like(a_arr)
        for i, ai in enumerate(a_arr):
            Om_ai = _Om_of_a(ai, Om0, w0, wa)
            H2[i] = Om_ai * ai**(-3)  # H^2/H0^2 = Om(a)*a^-3 for flat
        
        # q(a) = 0.5 * d ln H2 / d ln a = 0.5 * a * dH2/da / H2
        # Numerical derivative
        q_arr = np.zeros_like(a_arr)
        for i in range(1, n_steps-1):
            da = a_arr[i+1] - a_arr[i-1]
            dH2 = H2[i+1] - H2[i-1]
            q_arr[i] = 0.5 * a_arr[i] * dH2 / (da * H2[i])
        q_arr[0] = q_arr[1]
        q_arr[-1] = q_arr[-2]
        
        # Solve ODE: D' = v, v' = -(3/a - q(a))v + (3/2)Om(a)D
        # Use RK4 with small steps
        D = np.zeros_like(a_arr)
        v = np.zeros_like(a_arr)
        
        # Boundary condition at a=1: D(1)=1, v(1)=f(1)
        # We need to determine f(1). For LCDM, f(1)≈0.55
        # For general CPL, f(1) can be estimated
        # Let's use f(1) = 0.55 * Om0^0.55 as initial guess, then refine
        
        # Actually, for a proper reference, I should integrate from early times
        # with D(a) ~ a at early times (matter domination)
        
        # Let me use a shooting method: integrate from a=0.01 where D~a
        D[0] = a_arr[0]
        v[0] = 1.0  # D' = 1 in matter domination
        
        dt = a_arr[1] - a_arr[0]
        for i in range(n_steps - 1):
            ai = a_arr[i]
            Di = D[i]
            vi = v[i]
            qi = q_arr[i]
            Omi = _Om_of_a(ai, Om0, w0, wa)
            
            def rhs(D_val, v_val):
                D_prime = v_val
                v_prime = -(3.0/ai - qi) * v_val + (1.5 * Omi) * D_val
                return D_prime, v_prime
            
            k1_D, k1_v = rhs(Di, vi)
            k2_D, k2_v = rhs(Di + 0.5*dt*k1_D, vi + 0.5*dt*k1_v)
            k3_D, k3_v = rhs(Di + 0.5*dt*k2_D, vi + 0.5*dt*k2_v)
            k4_D, k4_v = rhs(Di + dt*k3_D, vi + dt*k3_v)
            
            D[i+1] = Di + (dt/6.0) * (k1_D + 2*k2_D + 2*k3_D + k4_D)
            v[i+1] = vi + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        # Normalize so D(1) = 1
        D_norm = D / D[-1]
        v_norm = v / D[-1]
        
        # Interpolate to find D(a_eval) and f(a_eval) = v/D
        idx = np.searchsorted(a_arr, a_eval)
        if idx <= 0:
            return D_norm[0], v_norm[0] / D_norm[0]
        if idx >= n_steps:
            return D_norm[-1], v_norm[-1] / D_norm[-1]
        
        t = (a_eval - a_arr[idx-1]) / (a_arr[idx] - a_arr[idx-1])
        D_val = D_norm[idx-1] + t * (D_norm[idx] - D_norm[idx-1])
        v_val = v_norm[idx-1] + t * (v_norm[idx] - v_norm[idx-1])
        f_val = v_val / D_val
        return D_val, f_val
    
    D_ref, f_ref = _solve_growth_ODE(a, Om0, w0, wa)
    return f_ref, D_ref

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline for safety
    
    # Test cosmologies (w0, wa, Om0)
    test_models = [
        (-1.0, 0.0, 0.3),      # LCDM
        (-1.0, 0.1, 0.3),      # CPL with wa=0.1
        (-0.9, 0.0, 0.3),      # CPL with w0=-0.9
        (-1.1, 0.0, 0.3),      # CPL with w0=-1.1
        (-1.0, 0.3, 0.3),      # CPL with wa=0.3
        (-0.8, -0.2, 0.3),     # Mixed
        (-1.2, 0.5, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.27),     # LCDM with different Om
        (-0.95, 0.1, 0.28),    # Near-LCDM CPL
        (-1.05, -0.1, 0.32),   # Another near-LCDM
    ]
    
    # a values to check
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    def objective(coeffs):
        """Max relative error over all test models and a values."""
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            for a in a_values:
                # Compute our approximation
                f_approx = _f_of_a(a, coeffs, Om0, w0, wa)
                D_approx = _D_normalized(a, coeffs, Om0, w0, wa)
                
                # Compute reference
                f_ref, D_ref = _reference_f(a, Om0, w0, wa)
                
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
    
    # Try Nelder-Mead optimization
    best_coeffs = x0.copy()
    best_err = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
        
        # Check deadline
        if time.time() > deadline:
            return best_coeffs.tolist()
        
        # Try a few more local optimizations from different starting points
        starting_points = [
            [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        ]
        
        for sp in starting_points:
            if time.time() > deadline - 5.0:
                break
            sp_arr = np.array(sp)
            res = minimize(
                objective,
                sp_arr,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if res.fun < best_err:
                best_coeffs = res.x
                best_err = res.fun
        
    except Exception:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
