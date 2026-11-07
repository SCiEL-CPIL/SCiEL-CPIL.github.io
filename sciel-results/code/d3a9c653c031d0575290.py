"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0' = y1, y1' = -y0 over t_span.
    
    Strategy: Use a high-order explicit Runge-Kutta method (Dormand-Prince 5(4)
    or similar) with adaptive step control to achieve high accuracy with
    minimal function calls. Since this is a specific known problem (harmonic
    oscillator), we can also exploit the analytic solution if needed, but the
    contract requires f to be called at least once and counts calls.
    
    The scoring is: score = -log10(rel) - 0.5 * log10(calls)
    where rel = max(||y_end - y_ref|| / max(||y_ref||, 1e-12), 1e-14)
    
    To maximize score, we want:
    1. Very small rel (high accuracy)
    2. Very few calls to f
    
    For a linear harmonic oscillator, we can use a very high-order method or
    exploit the structure. However, we must call f at least once.
    
    Best approach: Use a single large step with a very high-order method, or
    use adaptive RK with tight tolerances but efficient step selection.
    
    Actually, since the problem is linear and we know the structure, we can
    use the fact that the solution is y0(t) = cos(t), y1(t) = -sin(t).
    But we still need to call f.
    
    Let's use a high-order fixed-step RK method. For a smooth problem like
    this, a high-order method with a moderate number of steps will be very
    accurate.
    
    Alternatively, we can use scipy's solve_ivp with DOP853 and tight tolerances,
    but that might use many function calls.
    
    Let's try: Call f once to verify it works, then use the analytic solution
    if we detect the harmonic oscillator structure. But that's risky.
    
    Safer: Use a high-order explicit RK method with adaptive stepping.
    Let's implement Dormand-Prince 5(4) with adaptive step control.
    
    Actually, for maximum efficiency: use a single high-order method.
    DOP853 is 8th order, uses 12 function evaluations per step.
    With just a few steps, we can get very high accuracy.
    
    Let's use scipy.integrate.solve_ivp with DOP853 and very tight tolerances,
    but limit the number of steps to keep calls low.
    
    Wait - the contract says "f must be called at least once" and counts all calls.
    solve_ivp will call f multiple times internally.
    
    Best strategy: Implement a high-order RK method manually with adaptive steps,
    targeting very high accuracy (rel < 1e-14) with minimal calls.
    
    For the harmonic oscillator, the solution is smooth and bounded. A high-order
    method with ~10-20 steps should achieve machine precision.
    
    Let's use DOP853 (8th order, 12 stages) with adaptive step control.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250  # 250s out of 330s wall-clock limit
    
    # Call f at least once
    _ = f(t0, y0)
    
    # Use DOP853 coefficients (8th order, 12 stages)
    # But implementing full adaptive DOP853 is complex.
    # Instead, let's use a simpler approach: fixed-step high-order RK.
    
    # For a harmonic oscillator with t in [0, 10], the solution is smooth.
    # A 6th-order RK method with ~50 steps should be extremely accurate.
    # Each step uses 6 function calls, so 50 steps = 300 calls.
    # That's very few calls and should give high accuracy.
    
    # Let's use a 6th-order Runge-Kutta method (e.g., 6(5) pair) with fixed steps.
    # Or even simpler: use RK4 with many steps.
    
    # RK4 with 1000 steps: 4000 calls, error ~ (h^4) * 1000 ~ very small
    # For h = 10/1000 = 0.01, local error ~ h^5, global error ~ h^4 * T
    # h^4 = 1e-8, so global error ~ 1e-7. That's rel ~ 1e-7, score ~ 7 - 0.5*log10(4000) ~ 7 - 3.2 = 3.8
    
    # To get rel ~ 1e-14, we need h^4 * T ~ 1e-14, so h ~ (1e-14/10)^(1/4) ~ 1e-3.5 ~ 3e-4
    # steps ~ 10/3e-4 ~ 33000, calls ~ 132000. Score ~ 14 - 0.5*log10(132000) ~ 14 - 4.4 = 9.6
    
    # But we can do better with higher-order methods.
    # 8th order RK: global error ~ h^8 * T
    # For h = 0.1 (100 steps), h^8 = 1e-8, global error ~ 1e-7. Still not enough.
    # For h = 0.01 (1000 steps), h^8 = 1e-16, global error ~ 1e-15. 
    # 1000 steps * 12 calls = 12000 calls. Score ~ 15 - 0.5*log10(12000) ~ 15 - 4.3 = 10.7
    
    # Let's use a high-order method. Implementing DOP853 is complex but doable.
    # Alternatively, use scipy's solve_ivp with DOP853 and tight tolerances.
    
    # Let's try scipy approach first - it's well-tested.
    from scipy.integrate import solve_ivp
    
    # Use DOP853 with very tight tolerances
    # rtol=1e-13, atol=1e-15 should give rel ~ 1e-14 or better
    # The number of calls depends on the adaptive stepper.
    
    # But we already called f once above. solve_ivp will call f more times.
    # We need to count all calls. Let's wrap f to count calls.
    
    call_count = [1]  # Already called once
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Use solve_ivp with DOP853
    # max_step to control accuracy
    result = solve_ivp(
        f_counted,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        max_step=(t1 - t0) / 100  # At least 100 steps
    )
    
    if not result.success:
        # Fallback to RK4 with many steps
        steps = 50000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f_counted(t, y)
            k2 = f_counted(t + h / 2, y + h / 2 * k1)
            k3 = f_counted(t + h / 2, y + h / 2 * k2)
            k4 = f_counted(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    return result.y[:, -1]

# EVOLVE-BLOCK-END
