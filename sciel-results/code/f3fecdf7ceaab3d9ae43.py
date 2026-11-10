"""Growth-index compact approximation with smooth optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # --- Reference ODE for linear growth in CPL dark energy ---
    def omega_m(a, Om0, w0, wa):
        """Matter density at scale factor a."""
        if a == 0:
            return np.inf
        x = a ** (-3)
        y = a ** (-(3.0 + 3.0 * w0 + 3.0 * wa))
        # w_eff integral: ∫0^a a'^(-4-3w) da' / ∫0^1 ...
        # For CPL: rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * (1+wa)^(1-a)
        # Use the standard form: Omega_de(a) = Omega_de0 * a^(-3(1+w0)) * (1+wa)^(1-a) / E(a)^2
        # where E^2 = Om0*a^-3 + Ode0 * a^(-3(1+w0)) * (1+wa)^(1-a)
        Ode0 = 1.0 - Om0
        if Ode0 <= 0:
            return 1.0
        # de factor
        if wa == 0:
            de_factor = Ode0 * a ** (-3.0 * (1.0 + w0))
        else:
            # (1+wa)^(1-a) * a^(-3(1+w0+wa)) / (1+wa)  -- standard CPL form
            # rho_de(rho_de0) = rho_de0 * a^(-3(1+w0)) * (1+wa)^(1-a)
            # So Omega_de(a) = Ode0 * a^(-3(1+w0)) * (1+wa)^(1-a) / E(a)^2
            de_factor = Ode0 * a ** (-3.0 * (1.0 + w0)) * (1.0 + wa) ** (1.0 - a)
        E2 = Om0 * x + de_factor
        if E2 <= 0:
            return 1.0
        return Om0 * x / E2

    def solve_growth(Om0, w0, wa, a_vals):
        """Solve D'' + 2(H + H'/2H)D' - (3/2)Om(a)H^2 D = 0.
        Returns D normalized so D(1)=1, and f = dlnD/dlna."""
        # Use log-space integration: let y = ln(D), then
        # y'' + (2 + H'/H) y' - (3/2) Om(a) = 0  (in terms of ln a)
        # Actually: D'' + 2(1+q) D' - (3/2) Om(a) D = 0 where q = (1/2)Om - 1
        # In x = ln a: d2D/dx2 + 2(1+q) dD/dx - (3/2) Om(a) D = 0
        # Let f = dD/dx / D = dlnD/dx (growth rate)
        # f' + 2(1+q) f - (3/2) Om(a) = 0  ... wait, let me redo.
        # D''/D = -2(1+q) D'/D + (3/2) Om(a)
        # f = D'/D, f' = D''/D - (D'/D)^2 = -2(1+q)f + (3/2)Om - f^2
        # So f' = -f^2 - 2(1+q) f + (3/2) Om(a)
        # And D'/D = f, so D = exp(integral f dx)
        # Normalize D(1)=1: D(a)/D(1) = exp(-int_a^1 f dx)
        
        # We need to integrate from a small value to 1.
        # Use scipy's solve_ivp or do RK4 manually.
        from scipy.integrate import solve_ivp
        
        a_min = 0.01
        a_max = 1.0
        # Grid for integration
        n_int = 2000
        a_grid = np.linspace(a_min, a_max, n_int)
        
        def rhs(x, y):
            a = np.exp(x)
            Om = omega_m(a, Om0, w0, wa)
            q = 0.5 * Om - 1.0
            f = y[0]
            D_log = y[1]
            f_prime = -f**2 - 2.0*(1.0+q)*f + 1.5*Om
            D_log_prime = f
            return [f_prime, D_log_prime]
        
        # Initial conditions at a_min: use the known asymptotic f ≈ Om(a)^0.55
        a_init = a_min
        Om_init = omega_m(a_init, Om0, w0, wa)
        f_init = Om_init ** 0.55
        D_log_init = 0.0  # will normalize later
        
        x_init = np.log(a_init)
        x_end = np.log(a_max)
        
        # Evaluate at desired a_vals
        a_eval = np.clip(a_vals, a_min, a_max)
        t_eval = np.log(a_eval)
        
        sol = solve_ivp(rhs, [x_init, x_end], [f_init, D_log_init], 
                        t_eval=t_eval, rtol=1e-10, atol=1e-12, method='DOP853')
        
        if not sol.success:
            return None, None
        
        f_ref = sol.y[0]
        D_log_ref = sol.y[1]
        
        # Normalize D so D(1) = 1
        # D(a)/D(1) = exp(D_log(a) - D_log(1))
        # Find index closest to a=1
        idx_1 = np.argmin(np.abs(a_eval - 1.0))
        D_log_1 = D_log_ref[idx_1]
        D_norm = np.exp(D_log_ref - D_log_1)
        
        return f_ref, D_norm

    # --- Gamma and f from coefficients ---
    def compute_gamma_f(c, a, Om0, w0, wa):
        """Compute gamma and f from coefficients at scale factor a."""
        Om = omega_m(a, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        c0, c1, c2, c3, c4, c5 = c
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        # Clamp gamma to valid range for safety
        gamma = np.clip(gamma, 0.1, 1.2)
        f = Om ** gamma
        return gamma, f

    # --- Objective: max relative error over all test points ---
    # Validation cosmologies (CPL parameters)
    Om0_list = [0.25, 0.27, 0.30, 0.35]
    w0_list = [-1.0, -0.9, -0.8, -0.7]
    wa_list = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3]
    
    # Scale factors to check
    a_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Precompute reference solutions
    print("Computing reference solutions...", flush=True)
    refs = {}
    for Om0 in Om0_list:
        for w0 in w0_list:
            for wa in wa_list:
                key = (Om0, w0, wa)
                f_ref, D_ref = solve_growth(Om0, w0, wa, a_test)
                if f_ref is not None:
                    refs[key] = (f_ref, D_ref)
    
    print(f"Computed {len(refs)} reference solutions", flush=True)
    
    # --- Smooth max objective (log-sum-exp) ---
    def objective(c, smooth_beta=50.0):
        """Return smooth approximation to 100 * max relative error."""
        if time.time() - t0 > DEADLINE:
            return np.inf
        c = np.array(c, dtype=float)
        # Check coefficient bounds
        if np.any(c < -10) or np.any(c > 10):
            return 1e10
        
        errors = []
        for key, (f_ref, D_ref) in refs.items():
            Om0, w0, wa = key
            # Compute our f and D
            Om_arr = np.array([omega_m(a, Om0, w0, wa) for a in a_test])
            q_arr = 1.0 - Om_arr
            u_arr = 1.0 + w0 + wa * (1.0 - a_test)
            c0, c1, c2, c3, c4, c5 = c
            gamma_arr = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa*a_test + c5*u_arr*q_arr
            
            # Check gamma constraint
            if np.any(gamma_arr < 0.1 - 1e-6) or np.any(gamma_arr > 1.2 + 1e-6):
                # Penalty for constraint violation
                viol = np.max(np.abs(gamma_arr - np.clip(gamma_arr, 0.1, 1.2)))
                return 1e10 + viol
            
            f_our = Om_arr ** gamma_arr
            D_our = np.exp(-np.trapz(f_our, a_test[::-1], axis=0))  # This is wrong, need proper integration
            
            # Proper D: D(a)/D(1) = exp(-int_a^1 f(exp(x)) dx) where x=ln a
            # int_a^1 f(exp(x)) dx = int_{ln a}^0 f(exp(x)) dx
            # Use numerical integration
            D_our = np.zeros_like(a_test)
            for i, ai in enumerate(a_test):
                if ai >= 1.0:
                    D_our[i] = 1.0
                else:
                    # Integrate f from x=ln(ai) to x=0
                    x_grid = np.linspace(np.log(ai), 0.0, 200)
                    a_grid = np.exp(x_grid)
                    Om_grid = np.array([omega_m(a, Om0, w0, wa) for a in a_grid])
                    q_grid = 1.0 - Om_grid
                    u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
                    gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
                    f_grid = Om_grid ** gamma_grid
                    integral = np.trapz(f_grid, x_grid)
                    D_our[i] = np.exp(-integral)
            
            # Relative errors
            f_err = np.abs(f_our - f_ref) / np.abs(f_ref + 1e-30)
            D_err = np.abs(D_our - D_ref) / np.abs(D_ref + 1e-30)
            errors.extend(f_err.tolist())
            errors.extend(D_err.tolist())
        
        errors = np.array(errors)
        # Smooth max via log-sum-exp
        max_err = np.max(errors)
        if smooth_beta > 1:
            scaled = smooth_beta * errors
            scaled = scaled - np.max(scaled)
            log_sum_exp = np.max(scaled) + np.log(np.sum(np.exp(scaled)))
            smooth_max = log_sum_exp / smooth_beta
        else:
            smooth_max = max_err
        
        return 100.0 * smooth_max

    # --- Optimization ---
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = objective(best_c)
    print(f"Initial: {best_val:.6f}", flush=True)
    
    # Gradient descent with numerical gradients
    lr = 0.01
    n_iter = 200
    eps = 1e-4
    
    for it in range(n_iter):
        if time.time() - t0 > DEADLINE - 5:
            break
        c = best_c.copy()
        # Compute gradient numerically
        grad = np.zeros(6)
        for i in range(6):
            c_plus = c.copy()
            c_plus[i] += eps
            c_minus = c.copy()
            c_minus[i] -= eps
            fp = objective(c_plus, smooth_beta=20.0)
            fm = objective(c_minus, smooth_beta=20.0)
            grad[i] = (fp - fm) / (2 * eps)
        
        # Adaptive step size
        step = lr / (np.linalg.norm(grad) + 1e-8)
        c_new = c - step * grad
        # Project to bounds
        c_new = np.clip(c_new, -10, 10)
        
        val_new = objective(c_new, smooth_beta=20.0)
        if val_new < best_val:
            best_val = val_new
            best_c = c_new
            if it % 20 == 0:
                print(f"Iter {it}: {best_val:.6f}", flush=True)
        else:
            # Reduce learning rate
            lr *= 0.9
    
    # Final polish with exact max
    print(f"After GD: {best_val:.6f}", flush=True)
    if time.time() - t0 < DEADLINE - 2:
        # Try a few random restarts around best
        for _ in range(5):
            if time.time() - t0 > DEADLINE - 2:
                break
            c_try = best_c + np.random.normal(0, 0.01, 6)
            c_try = np.clip(c_try, -10, 10)
            val_try = objective(c_try, smooth_beta=1.0)  # exact max
            if val_try < best_val:
                best_val = val_try
                best_c = c_try
    
    result = [float(x) for x in best_c]
    print(f"Final: {best_val:.6f}, coeffs={result}", flush=True)
    return result
# EVOLVE-BLOCK-END
