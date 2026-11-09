"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin

    # Strategy: Use DOP853 (an 8th-order explicit Runge-Kutta method)
    # With a sufficiently small step size, the local truncation error is negligible
    # for the range [0, 10], achieving rel ~ 1e-14.
    # DOP853 requires 12 function evaluations per step.
    # We want calls < 2,000,000.
    # Let's target ~1000 steps. 1000 * 12 = 12000 calls.
    # This is well within the limit and fast.
    # The error for DOP853 with h=0.01 over t=10 is extremely small.
    
    # To be safe on accuracy, we can use more steps if time permits, 
    # but 1000 steps of DOP853 is already super-accurate for this problem.
    # Let's start with 1000 steps.
    
    n_steps = 1000
    h = (t1 - t0) / n_steps
    
    # DOP853 Butcher tableau
    # c = [0, 1/40, 1/10, 1/5, 1/4, 3/10, 1/3, 1/2, 5/8, 1, 1, 1]
    # A = 
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 1/40 0 0 0 0 0 0 0 0 0 0 0
    # 0 5/32 3/32 0 0 0 0 0 0 0 0 0
    # 0 0 184/255 0 0 0 0 0 0 0 0 0
    # 0 0 14/17 0 0 0 0 0 0 0 0 0
    # 0 0 0 4/5 0 0 0 0 0 0 0 0
    # 0 0 0 0 1/2 0 0 0 0 0 0 0
    # 0 0 0 0 0 1/2 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 25/24 0 0 0 0
    # 0 0 0 0 0 0 0 0 1 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 0 0
    # 0 0 0 0 0 0 0 0 0 0 1 0
    
    # b = [35/384, 0, 500/1113, 125/192, 2187/6784, 11/84, 82/249, 0, 0, 5129179/564318, 7571/13824, 0]
    # (Note: The last stage is for error estimation in embedded methods, but for pure DOP853 step, 
    # we use the 8th order weights. Standard DOP853 has 12 stages, 11th and 12th are often 
    # used for error estimation in DOP853(2,7) or similar, but the 8th order solution 
    # uses specific weights. Let's use a standard high-order RK to be safe and simple, 
    # or just use SciPy's solve_ivp if allowed? 
    # "Python standard library, NumPy and SciPy are available."
    # Using SciPy's solve_ivp is the most robust and accurate way, and it handles the 
    # call counting naturally if we just count the function calls? 
    # Wait, the task says "f must be called at least once" and "every call is counted".
    # If I use scipy.integrate.solve_ivp, it will call f. I need to count those calls.
    # But I can't easily intercept the calls made inside solve_ivp unless I wrap f.
    
    # Let's wrap f to count calls.
    call_count = 0
    
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Use solve_ivp with DOP853, tight tolerances.
    # This is likely the best strategy: let the library do the heavy lifting with 
    # adaptive stepping, which is highly efficient for smooth problems like the 
    # harmonic oscillator.
    
    try:
        from scipy.integrate import solve_ivp
        
        sol = solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            dense_output=False
        )
        
        if not sol.success:
            # Fallback if solve_ivp fails
            pass
        
        # Check time
        if time.time() > deadline:
            return sol.y[:, -1] if sol.t.size > 0 else y0

        if sol.success:
            return sol.y[:, -1]
            
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: Simple RK4 with many steps if SciPy fails or is too slow
    # (Unlikely to be needed, but safe)
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0.copy()
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
            
        k1 = f_wrapper(t, y)
        k2 = f_wrapper(t + h / 2, y + h / 2 * k1)
        k3 = f_wrapper(t + h / 2, y + h / 2 * k2)
        k4 = f_wrapper(t + h, y + h * k3)
        
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y

# EVOLVE-BLOCK-END
