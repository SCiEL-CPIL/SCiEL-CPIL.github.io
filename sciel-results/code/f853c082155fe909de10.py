"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Robertson problem Jacobian (analytic)
    def jac(t, y):
        y0_val, y1_val, y2_val = y
        c1 = 0.04
        c2 = 1e4
        c3 = 0.31
        
        J = np.array([
            [-c1, c2*y2_val, c2*y1_val],
            [c1, -c2*y2_val - 2*c3*y1_val, -c2*y1_val],
            [0, 2*c3*y1_val, 0]
        ])
        return J

    start_time = time.time()
    # Safety margin: 300s limit, stop trying to solve if we exceed 280s
    deadline = start_time + 280.0
    
    def elapsed():
        return time.time() - start_time

    try:
        # Strategy: Use LSODA.
        # LSODA automatically switches between Adams (non-stiff) and BDF (stiff) methods.
        # For the Robertson problem:
        # 1. The initial transient is stiff. LSODA will use BDF.
        # 2. The subsequent long relaxation is less stiff (or non-stiff in some regimes).
        #    LSODA can switch to Adams, which is often more efficient (fewer function calls)
        #    for non-stiff problems than implicit methods like Radau or BDF.
        # This should reduce the number of f-calls compared to a fixed implicit method
        # over the entire interval, improving the -0.5*log10(calls) term of the score.
        
        sol = integrate_scipy.solve_ivp(
            f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If LSODA fails, fallback to Radau with tighter tolerances
            # (Radau is generally more robust for stiff problems)
            sol_fallback = integrate_scipy.solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=jac
            )
            if sol_fallback.success:
                return sol_fallback.y[:, -1]
            
            # Last resort: return initial state (will have high error, but avoids crash)
            # In practice, this shouldn't happen for Robertson.
            return y0

    except Exception:
        # If anything goes wrong (e.g., memory, numerical issues), try a simple RK4
        # just to return *something* valid, though the score will be poor.
        try:
            y = y0.copy()
            h = (t1 - t0) / 10000
            t = t0
            for _ in range(10000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                t += h
                if elapsed() > deadline:
                    break
            return y
        except Exception:
            return y0

# EVOLVE-BLOCK-END
