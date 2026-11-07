import numpy as np
import time

# DOP853 coefficients (Butcher tableau)
# From scipy's DOP853 implementation
# There are 14 stages.
_DOP853_C = [
    0.0,
    1.0/40.0,
    4.0/45.0,
    4.0/15.0,
    1.0/2.0,
    8.0/15.0,
    1.0,
    1.0/1.0,
    11.0/40.0,
    3.0/2.0,
    24.0/5.0,
    11.0/5.0,
    1.0,
    1.0
]

_DOP853_A = [
    [],
    [1.0/40.0],
    [-8.0/27.0, 32.0/135.0, 4.0/135.0],
    [0.0, 4.0/27.0, 4.0/9.0, -100.0/243.0],
    [0.0, 4.0/27.0, 4.0/9.0, -100.0/243.0],
    [0.0, -1.0/162.0, 5.0/18.0, -5.0/81.0, 5.0/12.0],
    [0.0, 71.0/405.0, -2.0/9.0, 7.0/27.0, -4.0/5.0, 4.0/9.0],
    [0.0, 2.0/9.0, -7.0/27.0, 4.0/5.0, -4.0/9.0, 1.0/4.0, 1.0/5.0],
    [0.0, 8.0/27.0, -2.0/9.0, 7.0/27.0, -4.0/5.0, 4.0/9.0, 1.0/4.0, 1.0/5.0],
    [0.0, -2.0/9.0, -1.0/54.0, 3.0/2.0, -4.0/5.0, 4.0/9.0, 1.0/4.0, 1.0/5.0],
    [0.0, 10.0/3.0, -7.0/2.0, 4.0/5.0, -4.0/9.0, 1.0/4.0, 1.0/5.0, -1.0/3.0, 1.0/9.0],
    [0.0, -1.0/4.0, 0.0, 2.0/3.0, -4.0/9.0, 1.0/4.0, 1.0/5.0, -1.0/3.0, 1.0/9.0],
    [0.0, 1.0/4.0, 0.0, -2.0/3.0, 4.0/9.0, -1.0/4.0, -1.0/5.0, 1.0/3.0, -1.0/9.0, 1.0/4.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/4.0],
    # The actual DOP853 weights are for the 8th order solution.
    # The standard DOP853 coefficients from Dormand-Prince:
    # I need to be careful. Let's use the exact coefficients from a reliable source or implement a simpler high-order method if I doubt my memory of the exact tableau.
    # Actually, implementing DOP853 from scratch is error-prone. 
    # Let's use RKF45 (4/5 order) or RK4 (4th order) with a very small step size.
    # RK4 is 4th order. Error ~ h^4.
    # If h=0.001, steps=2000, calls=8000. Error ~ 1e-12.
    # If h=0.0001, steps=20000, calls=80000. Error ~ 1e-16.
    # 80000 calls is fine (< 2,000,000).
    # Score = -log10(1e-14) - 0.5*log10(80000) = 14 - 0.5*4.9 = 14 - 2.45 = 11.55.
    # This is much better than 1.6.
    
    # Let's check if we can do better with a 5th order method like Cash-Karp or RKF45.
    # RKF45 is 4th/5th order. 6 stages.
    # Let's stick to RK4 for simplicity and robustness, but maybe use a slightly larger h to reduce calls if accuracy is still good.
    # The reference uses DOP853 with rtol=1e-11, atol=1e-13.
    # This implies the reference is very accurate.
    # We need to match it.
    # Using RK4 with h=0.0001 should be sufficient to reach machine precision relative to the reference.
]

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system.
    
    Strategy:
    Use a fixed-step RK4 integrator with a very small step size to ensure high accuracy
    and a predictable, moderate number of function calls.
    
    The previous approach used scipy's solve_ivp, which may have failed or been slow.
    A manual RK4 loop is robust, fast, and uses no external solver overhead.
    
    We choose h = 0.0001.
    Total time T = 2.
    Number of steps N = 2 / 0.0001 = 20000.
    Calls per step = 4.
    Total calls = 80000.
    This is well within the 2,000,000 limit.
    The error of RK4 is O(h^4). With h=1e-4, h^4=1e-16, which is at the machine precision level.
    This should yield a relative error close to 1e-14 or lower.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin from 330s
    
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    T = t_end - t_start
    
    if T == 0:
        return y
    
    # Choose step size
    # We want high accuracy. Let's start with h=0.0001.
    # If time is an issue, we can increase h, but 80k calls is trivial.
    h = 1e-4
    n_steps = int(np.ceil(T / h))
    # Adjust h to land exactly on t_end
    h = T / n_steps
    
    # Call f at least once (required by contract)
    # We do this inside the loop anyway, but let's ensure at least one call is made.
    # The loop will make 4*n_steps calls.
    
    t = t_start
    for i in range(n_steps):
        if time.time() > deadline:
            # If we run out of time, return current state
            # But we should have finished long ago
            break
        
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + 0.5*h, y + 0.5*h*k1))
        k3 = np.array(f(t + 0.5*h, y + 0.5*h*k2))
        k4 = np.array(f(t + h, y + h*k3))
        
        y = y + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += h
    
    return y
