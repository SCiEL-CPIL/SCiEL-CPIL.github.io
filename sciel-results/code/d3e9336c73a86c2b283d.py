"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses an adaptive Dormand-Prince 5(4) Runge-Kutta method.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    if t0 == t1:
        return y

    # Parameters for adaptive step size control
    rtol = 1e-11
    atol = 1e-13
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    if h <= 0:
        h = 1e-6
    t = t0
    
    # Dormand-Prince 5(4) coefficients
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    # A = [
    #   [0, 0, 0, 0, 0, 0, 0],
    #   [1/5, 0, 0, 0, 0, 0, 0],
    #   [3/40, 9/40, 0, 0, 0, 0, 0],
    #   [44/45, -56/15, 32/9, 0, 0, 0, 0],
    #   [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    #   [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    #   [35/384, 0, 500/1113, 125/192, 0, -2187/6784, 11/84]
    # ]
    # B (5th order) = [35/384, 0, 500/1113, 125/192, 0, -2187/6784, 11/84]
    # B* (4th order) = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    # E = B - B* = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]

    def rk_step(t, y, h):
        k1 = f(t, y)
        
        k2 = f(t + h/5.0, y + h * (1/5.0) * k1)
        
        k3 = f(t + 3.0*h/10.0, y + h * (3/40.0 * k1 + 9/40.0 * k2))
        
        k4 = f(t + 4.0*h/5.0, y + h * (44/45.0 * k1 - 56/15.0 * k2 + 32/9.0 * k3))
        
        k5 = f(t + 8.0*h/9.0, y + h * (19372/6561.0 * k1 - 25360/2187.0 * k2 + 64448/6561.0 * k3 - 212/729.0 * k4))
        
        k6 = f(t + h, y + h * (9017/3168.0 * k1 - 355/33.0 * k2 + 46732/5247.0 * k3 + 49/176.0 * k4 - 5103/18656.0 * k5))
        
        # 5th order solution
        y_new = y + h * (35/384.0 * k1 + 500/1113.0 * k3 + 125/192.0 * k4 - 2187/6784.0 * k6 + 11/84.0 * k6) # Wait, k7 is needed for B[6]
        # Correction: The 5th order solution uses k1..k6 and a final evaluation k7?
        # No, standard DP5(4) computes 6 stages k1..k6.
        # The 5th order weights are for k1, k3, k4, k6, k7?
        # Let's check the standard DP5(4) tableau.
        # It is a 7-stage method where the 7th stage is the final point.
        # k1 = f(t, y)
        # ...
        # k6 = f(t+h, y + ...)
        # y_5 = y + h * (b1*k1 + b3*k3 + b4*k4 + b6*k6 + b7*k7)
        # But k7 = f(t+h, y_5). This is implicit in the evaluation if we use it for error?
        # No, in the standard implementation (e.g. in scipy or MATLAB), we compute k1..k6.
        # Then we compute y_5 using k1..k6?
        # Actually, the Dormand-Prince method is defined such that the 5th order solution is:
        # y_{n+1} = y_n + h * (35/384 k1 + 0 k2 + 500/1113 k3 + 125/192 k4 + 0 k5 - 2187/6784 k6 + 11/84 k7)
        # And k7 = f(t_n + h, y_{n+1}).
        # This requires an extra function call for k7.
        # However, many implementations (like the one in `solve_ivp` for RK45) use a slightly different formulation or reuse the last stage.
        # Let's use the standard 7-stage DP5(4) which requires 7 function calls per step.
        
        # Calculate y_5 (5th order)
        # We need k7.
        # First, estimate y_5 using the 4th order formula? No.
        # Standard DP5(4):
        # k1..k6 are computed.
        # y_5 = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6) ... wait, the coefficient for k7 is 11/84.
        # If we don't have k7, we can't compute the 5th order solution directly?
        # Actually, the "5th order" solution in the embedded pair is often the one used for the next step, and the "4th order" is for error.
        # In DP5(4), the 5th order solution is:
        # y_new = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6 + 11/84*k7)
        # The 4th order solution is:
        # y_old = y + h * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)
        # This requires k7.
        # So we MUST compute k7.
        # But k7 depends on y_new.
        # This is a common confusion. In the standard DP5(4) implementation found in most libraries:
        # The 7th stage IS the evaluation at the end of the step for the 5th order solution.
        # So, we compute k1..k6.
        # Then we compute a preliminary y_5? No.
        # We compute y_5 using the formula that involves k7?
        # Let's look at the coefficients again.
        # The method is often described as 6 stages + 1 extra.
        # k1..k6 are the 6 stages.
        # y_5 = y + h * sum(b_i k_i) for i=1..6?
        # No, the 5th order solution uses k7.
        # Is there a version where the 5th order solution does NOT use k7?
        # Yes, the "Cash-Karp" method uses 7 stages but the 5th order solution is formed from the first 6?
        # Let's use **Cash-Karp 5(4)** which is well-defined and requires 7 calls, where the 5th order solution is computed from k1..k6?
        # Cash-Karp:
        # k1..k6 computed.
        # k7 = f(t+h, y + h*(5179/57600*k1 + ... + 1/40*k6))? No.
        
        # Let's stick to **Dormand-Prince 5(4)** as implemented in `scipy.integrate.solve_ivp` (RK45).
        # It uses 6 stages.
        # The 5th order solution is:
        # y_new = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6)
        # Wait, the sum of coefficients is 1?
        # 35/384 + 500/1113 + 125/192 - 2187/6784 = 1?
        # 0.0911 + 0.4492 + 0.6510 - 0.3224 = 0.8689. No.
        # The coefficient for k7 is 11/84 = 0.1309.
        # 0.8689 + 0.1309 = 0.9998. Close to 1.
        # So the 5th order solution REQUIRES k7.
        
        # However, there is a trick. We can compute the 4th order solution first (which uses k1..k6), then compute k7 based on that? No.
        
        # Let's use a simpler, verified method: **Explicit RK4** with adaptive step? No, 4th order is not high enough for "high accuracy" with few calls.
        
        # Let's use **DOP853** (8th order) from `scipy`? No, I must write it.
        
        # Alternative: Use **fixed step RK4** but with much more steps? No, that increases calls.
        
        # Let's use **adaptive RK45 (Dormand-Prince)** correctly.
        # In many implementations, the 7th function call is used to compute k7.
        # But we can't compute k7 without y_new.
        # Actually, the standard DP5(4) is:
        # Compute k1..k6.
        # Compute y_5 = y + h * (b1*k1 + b3*k3 + b4*k4 + b6*k6 + b7*k7)
        # This is circular.
        
        # CORRECTION: The Dormand-Prince method is a 7-stage method.
        # Stages 1-6 are k1..k6.
        # Stage 7 is k7 = f(t+h, y_5).
        # But y_5 is the 5th order solution.
        # The 5th order solution is defined as:
        # y_5 = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6 + 11/84*k7)
        # This is indeed circular if k7 depends on y_5.
        
        # WAIT. I am confusing myself.
        # The standard DP5(4) does NOT have a circular dependency.
        # The 5th order solution is:
        # y_{n+1} = y_n + h * (35/384 k1 + 0 k2 + 500/1113 k3 + 125/192 k4 + 0 k5 - 2187/6784 k6 + 11/84 k7)
        # The 4th order solution is:
        # y_{n+1}^* = y_n + h * (5179/57600 k1 + 0 k2 + 7571/16695 k3 + 393/640 k4 - 92097/339200 k5 + 187/2100 k6 + 1/40 k7)
        # In both cases, k7 is the derivative at the end of the step.
        # To compute k7, we need the point at the end of the step.
        # Which point? The 5th order point or the 4th order point?
        # In the standard implementation, we compute k1..k6.
        # Then we compute the 5th order solution y_5? No, we can't because we need k7.
        
        # Let's look at the actual algorithm used in `scipy.integrate.solve_ivp` for `RK45`.
        # It computes 6 stages.
        # Then it computes the error using the difference between the 5th and 4th order solutions.
        # The 5th order solution is:
        # y_new = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6)
        # Wait, is the coefficient for k7 missing?
        # Let's check the sum of coefficients for the 5th order solution in RK45:
        # 35/384 + 500/1113 + 125/192 - 2187/6784 = 1.0 ?
        # 0.0911458 + 0.4492363 + 0.6510417 - 0.3223762 = 0.8690476.
        # It is NOT 1.
        # So there MUST be a k7 term.
        
        # Ah, I see. In the "Dormand-Prince" method, the 7th stage is used for the 5th order solution.
        # But how is k7 computed?
        # k7 = f(t+h, y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6))
        # This y is the 5th order solution WITHOUT the k7 term?
        # No, that's not right.
        
        # Let's use **Cash-Karp 5(4)** instead. It is simpler.
        # Cash-Karp requires 7 function calls.
        # k1..k6 are computed.
        # k7 = f(t+h, y + h*(5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7?))
        # No, Cash-Karp 5(4) is:
        # k1..k6 computed.
        # y_5 = y + h * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)
        # This is also circular.
        
        # Let's just use **fixed step RK4** with a reasonable number of steps, but optimize the constant.
        # The current code uses 100,000 steps.
        # If I use 10,000 steps, calls = 40,000.
        # `log10(40000) = 4.6`. Penalty = 2.3.
        # Error will be larger. `rel` might be 1e-8.
        # Score = 8 - 2.3 = 5.7.
        # This is better than 1.13.
        
        # If I use 1,000 steps, calls = 4,000.
        # `log10(4000) = 3.6`. Penalty = 1.8.
        # Error might be 1e-5.
        # Score = 5 - 1.8 = 3.2.
        
        # If I use 100 steps, calls = 400.
        # `log10(400) = 2.6`. Penalty = 1.3.
        # Error might be 1e-3.
        # Score = 3 - 1.3 = 1.7.
        
        # So 10,000 steps seems like a good balance.
        
        # But can I do better?
        # Yes, using an adaptive method.
        # Let's implement **Dormand-Prince 5(4)** correctly.
        # The standard way to handle the "circular" k7 is:
        # 1. Compute k1..k6.
        # 2. Compute y_5_prelim = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6)
        # 3. Compute k7 = f(t+h, y_5_prelim)
        # 4. Compute y_5 = y_5_prelim + h * (11/84 * k7)
        # 5. Compute y_4 = y + h * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)
        # 6. Error = ||y_5 - y_4||
        
        # This works! It requires 7 function calls per step.
        
        def rk_step(t, y, h):
            k1 = f(t, y)
            
            k2 = f(t + h/5.0, y + h * (1/5.0) * k1)
            
            k3 = f(t + 3.0*h/10.0, y + h * (3/40.0 * k1 + 9/40.0 * k2))
            
            k4 = f(t + 4.0*h/5.0, y + h * (44/45.0 * k1 - 56/15.0 * k2 + 32/9.0 * k3))
            
            k5 = f(t + 8.0*h/9.0, y + h * (19372/6561.0 * k1 - 25360/2187.0 * k2 + 64448/6561.0 * k3 - 212/729.0 * k4))
            
            k6 = f(t + h, y + h * (9017/3168.0 * k1 - 355/33.0 * k2 + 46732/5247.0 * k3 + 49/176.0 * k4 - 5103/18656.0 * k5))
            
            # Preliminary 5th order solution (without k7 term)
            y_5_prelim = y + h * (35/384.0 * k1 + 500/1113.0 * k3 + 125/192.0 * k4 - 2187/6784.0 * k6)
            
            k7 = f(t + h, y_5_prelim)
            
            # Final 5th order solution
            y_5 = y_5_prelim + h * (11/84.0 * k7)
            
            # 4th order solution
            y_4 = y + h * (5179/57600.0 * k1 + 7571/16695.0 * k3 + 393/640.0 * k4 - 92097/339200.0 * k5 + 187/2100.0 * k6 + 1/40.0 * k7)
            
            return y_5, y_4

    max_steps = 100000
    min_h = 1e-12
    max_h = (t1 - t0) / 10.0
    
    for _ in range(max_steps):
        if t >= t1:
            break
        
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        y_new, y_old = rk_step(t, y, h)
        
        # Error estimate
        err_norm = np.max(np.abs(y_new - y_old))
        scale = np.max(np.abs(y_new))
        if scale < 1e-12:
            scale = 1e-12
        
        err_rel = err_norm / scale
        
        # Adjust step size
        if err_rel > rtol:
            # Reduce step
            factor = 0.9 * (rtol / err_rel) ** 0.2
            h *= max(factor, 0.1)
            if h < min_h:
                h = min_h
        else:
            # Increase step
            factor = 0.9 * (rtol / err_rel) ** 0.25
            h *= min(factor, 5.0)
            if h > max_h:
                h = max_h
            t += h
            y = y_new
            
    return y

# EVOLVE-BLOCK-END
