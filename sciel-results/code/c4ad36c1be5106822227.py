"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Strategy: Use scipy's DOP853 with extremely tight tolerances to minimize
# the number of function calls while maximizing accuracy. DOP853 is a 7th-order
# method with an embedded 6th-order error estimator, making it highly efficient
# for achieving high precision with few steps.
#
# Key insight: The reference solution uses DOP853 with rtol=1e-11, atol=1e-13.
# By using the same method but with tighter tolerances (rtol=1e-13, atol=1e-15),
# we can achieve accuracy well below 1e-14 relative error (the floor) while
# keeping the number of function calls low. DOP853 requires 12 function evaluations
# per step, and with large step sizes possible due to high order, we can complete
# the integration [0, 2] in very few steps.
#
# Alternative approach: If we want to reduce calls further, we could use a
# single very large step with a high-order method, but that risks accuracy.
# Instead, we let the adaptive stepper choose step sizes, which will naturally
# take large steps where the solution is smooth.

try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _fallback_rk45_step(f, t, y, h):
    """
    Fallback: Dormand-Prince 5(4) adaptive step.
    Returns y_new (5th order) and y_err (error estimate).
    """
    a_dp = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, 355/33, -46732/5247, 49/176, -5103/18656, 0]
    ])
    b_dp = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    c_dp = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    b_err = b_dp - np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100])

    k = np.zeros((6, len(y)))
    k[0] = f(t, y)
    for i in range(1, 6):
        y_stage = y + h * np.dot(a_dp[i, :i], k[:i])
        k[i] = f(t + c_dp[i] * h, y_stage)
    
    y_new = y + h * np.dot(b_dp, k)
    y_err = h * np.dot(b_err, k)
    return y_new, y_err


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 295  # 295s safety margin for 300s limit
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    if _HAS_SCIPY:
        # Use scipy's DOP853 with very tight tolerances.
        # DOP853 is 7th order, so it can take large steps while maintaining accuracy.
        # We want rel error < 1e-14, so we set rtol and atol very tight.
        # 
        # The reference uses rtol=1e-11, atol=1e-13.
        # We use rtol=1e-13, atol=1e-15 to be well below the reference.
        #
        # max_step: We allow the stepper to take large steps. For t in [0,2],
        # a max_step of 0.1 means at most 20 steps, but DOP853 will likely
        # use even fewer due to its high order and the smoothness of Lorenz.
        # Actually, let's allow larger steps. max_step=0.5 means at most 4 steps.
        # But we need to be careful: too large a step might reduce accuracy.
        # Let's try max_step=0.2 (10 steps max) and see.
        
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='DOP853', 
                rtol=1e-13, 
                atol=1e-15,
                max_step=0.2  # Allow up to 10 steps; DOP853 may use fewer
            )
            if sol.success:
                # Check if we're within time limit
                if time.time() < deadline:
                    return sol.y[:, -1]
                else:
                    # Return whatever we have
                    return sol.y[:, -1]
            else:
                # Fallback if solve_ivp fails
                pass
        except Exception as e:
            # Fallback to manual integration
            pass
    
    # Fallback: Manual adaptive RK45 integration
    t = t0
    y = y0_arr.copy()
    h = (t1 - t0) / 100  # Initial step size
    
    rtol = 1e-13
    atol = 1e-15
    
    while t < t1:
        if time.time() > deadline:
            # Return current best guess
            break
        
        # Limit step size to not overshoot t1
        h = min(h, t1 - t)
        
        # Perform step
        y_new, y_err = _fallback_rk45_step(f, t, y, h)
        
        # Calculate error norm
        err_norm = np.max(np.abs(y_err) / (atol + rtol * np.abs(y)))
        
        if err_norm <= 1.0:
            # Step accepted
            y = y_new
            t = t + h
            
            # Adjust step size
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, t1 - t)
            
        else:
            # Step rejected
            if err_norm > 0:
                h = h * max(0.1, 0.9 * err_norm ** (-0.25))
            else:
                h = h * 0.1
    
    return y
# EVOLVE-BLOCK-END
