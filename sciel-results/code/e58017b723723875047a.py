"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Ensure f is called at least once
    _ = f(t0, y0_arr)
    
    deadline = time.time() + 320  # safety margin before 330s kill
    
    # Strategy: Use scipy.integrate.solve_ivp with DOP853.
    # The key insight from the observations is that using a specific max_step
    # can reduce the number of function calls while maintaining high accuracy.
    # The reference uses rtol=1e-11, atol=1e-13.
    # We want rel < 1e-14.
    # DOP853 is 8th order.
    
    try:
        from scipy.integrate import solve_ivp
        
        # Try a few configurations.
        # Configuration 1: Tighter tolerances, no max_step restriction (adaptive).
        # Configuration 2: Tighter tolerances, with a max_step that forces more steps but potentially better accuracy.
        # Configuration 3: Even tighter tolerances.
        
        configs = [
            {'rtol': 1e-13, 'atol': 1e-15, 'max_step': (t1 - t0) / 100},
            {'rtol': 1e-14, 'atol': 1e-16, 'max_step': (t1 - t0) / 50},
            {'rtol': 1e-12, 'atol': 1e-14, 'max_step': (t1 - t0) / 200},
            {'rtol': 1e-13, 'atol': 1e-15, 'max_step': (t1 - t0) / 500},
        ]
        
        best_y = None
        best_calls = float('inf')
        
        for cfg in configs:
            if time.time() > deadline:
                break
            try:
                sol = solve_ivp(f, (t0, t1), y0_arr, method='DOP853', 
                                rtol=cfg['rtol'], atol=cfg['atol'], 
                                max_step=cfg['max_step'], dense_output=False)
                if sol.success and sol.y.shape[1] > 0:
                    y_end = sol.y[:, -1]
                    if np.all(np.isfinite(y_end)):
                        # We don't know the reference, but we assume tighter is better.
                        # However, the score penalizes calls.
                        # Let's just pick the first successful one with reasonable accuracy.
                        # Actually, we should pick the one with the fewest calls that is accurate enough.
                        # Since we can't check accuracy against reference, we rely on the tight tolerances.
                        # Let's assume the first one (tightest rtol) is good.
                        # But wait, the observation said "picking the fewest-call successful run".
                        # So we should compare call counts.
                        if sol.t.size < best_calls: # sol.t.size is steps+1, roughly proportional to calls
                            best_calls = sol.t.size
                            best_y = y_end
            except Exception:
                pass
        
        if best_y is not None:
            return best_y
            
    except Exception:
        pass

    # Fallback: High-order fixed-step RK4 with many steps
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0_arr.copy()
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
