"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    deadline = time.time() + 290.0  # safety margin before 330s kill
    
    # The system is fixed: Lotka-Volterra with known parameters.
    # We can precompute the reference solution using a high-accuracy method.
    # But we don't know the reference exactly, so we need to be very accurate.
    
    # Strategy: Use a high-order adaptive Runge-Kutta method (Dormand-Prince or similar)
    # with tight tolerances to match DOP853 at rtol=1e-11, atol=1e-13.
    
    # Since f is counted, we want to minimize calls while maintaining accuracy.
    # A fixed-step high-order RK method with a reasonable number of steps should work.
    
    # Let's use an embedded Runge-Kutta method with adaptive step control.
    # Actually, for this specific non-stiff system over [0,5], a well-tuned fixed-step
    # high-order method should be sufficient.
    
    # Try RK45 (Dormand-Prince) with adaptive stepping, tight tolerances.
    # Or better: use a fixed-step 8th-order Runge-Kutta with enough steps.
    
    # Let's try: 8th order Runge-Kutta with adaptive step size control.
    # But to keep it simple and efficient, let's use a well-tested approach.
    
    # Approach: Implement RK45 (Dormand-Prince) with adaptive step size.
    # Tolerances: rtol=1e-12, atol=1e-14 to be tighter than the reference.
    
    # Dormand-Prince coefficients
    # Order 5(4) pair
    _a = [
        0, 1/5, 3/10, 4/5, 8/9, 1, 1
    ]
    _b = [
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ]
    _c4 = [25/216, 0, 1408/2565, 2197/4104, -1/10, 0, 1]
    _c5 = [1691/2160, 0, -1220/567, 2285/672, 68/105, -1/40, 1]
    
    # This is getting complex. Let me use scipy if available, but we can't import it in the function.
    # Wait, we can use scipy.integrate.solve_ivp but that would count f calls through our wrapper.
    # The problem says f is counted on every call. If we call solve_ivp with our f, it will count.
    
    # Better approach: Use a fixed high-order RK method with a sufficient number of steps.
    # For Lotka-Volterra over [0,5], the solution is bounded and smooth.
    # Let's estimate: frequencies are roughly sqrt(3)~1.73 and the system has period ~3.6.
    # So 5 units covers about 1.4 periods. A step of h=0.01 gives 500 steps.
    # With RK4 (4th order), error per step ~ h^5, total ~ h^4 = 1e-8. Not enough.
    # With h=0.001, 5000 steps, error ~ 1e-12. That might work with RK4.
    # But 5000*4 = 20000 f calls. Score: -log10(1e-12) - 0.5*log10(20000) = 12 - 0.5*4.3 = 12-2.15=9.85
    # Parent: RK4 with 2000 steps = 8000 calls. Error might be ~1e-10. Score: 10-0.5*3.9=10-1.95=8.05
    # So 5000 steps RK4 could be better.
    
    # But we can do better with higher-order methods.
    # Let's use an 8th-order Runge-Kutta method.
    
    # Actually, let me just use scipy's solve_ivp with DOP853 directly, but that's what the reference uses.
    # The f calls will be counted. DOP853 is 8th order, adaptive. For this smooth problem, it should take
    # relatively few steps.
    
    # But wait - can I call scipy.integrate.solve_ivp inside integrate? Yes, scipy is available.
    # The f function will be called by solve_ivp, and each call is counted.
    
    # Let me try this approach:
    try:
        from scipy.integrate import solve_ivp
        # Use tighter tolerances than reference to ensure our solution is more accurate
        # Reference uses DOP853 with rtol=1e-11, atol=1e-13
        # We use rtol=1e-13, atol=1e-15 to be 100x tighter
        sol = solve_ivp(f, (t0, t1), y0_arr, method='DOP853', rtol=1e-13, atol=1e-15, dense_output=False)
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Fallback: fixed-step RK4 with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0_arr.copy()
    for i in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            break
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
