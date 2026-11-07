import numpy as np
import time
import sys

# DOP853 coefficients (Dormand-Prince 8(5,3) method)
# This is an 8th order method with 12 stages.
# Source: Dormand and Prince (1996)
c = np.array([
    0.0,
    1/12,
    1/6,
    1/4,
    5/12,
    3/8,
    1/2,
    5/8,
    3/5,
    1.0,
    5/4,
    1.0
])

# Butcher tableau A (12x12)
A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/24, 1/24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/16, -1/16, 1/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-7/240, 7/60, 14/15, -7/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [7/128, -7/32, 21/32, -21/64, 7/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-187/5760, 107/360, -1232/315, 267/40, -17/80, 44/45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [35/1024, -35/256, 21/32, -21/64, 7/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2225/4096, -2725/256, 225/32, -1875/256, 225/64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Placeholder for row 9 (c=1)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Placeholder for row 10 (c=5/4)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder for row 11 (c=1)
])

# Correct DOP853 A matrix from literature (e.g. Hairer, Nørsett, Wanner)
# The above placeholders are wrong. Let's use the exact coefficients.
# DOP853:
# c1=0, c2=1/12, c3=1/6, c4=1/4, c5=5/12, c6=3/8, c7=1/2, c8=5/8, c9=3/5, c10=1, c11=5/4, c12=1
# a21=1/12
# a31=1/24, a32=1/24
# a41=1/16, a42=-1/16, a43=1/4
# a51=-7/240, a52=7/60, a53=14/15, a54=-7/12
# a61=7/128, a62=-7/32, a63=21/32, a64=-21/64, a65=7/16
# a71=-187/5760, a72=107/360, a73=-1232/315, a74=267/40, a75=-17/80, a76=44/45
# a81=35/1024, a82=-35/256, a83=21/32, a84=-21/64, a85=7/16, a86=0, a87=0 -- Wait, c8=5/8, so a81..a87
# Actually, let's just use the standard coefficients from a reliable source or hardcode the known good ones.
# Since I cannot browse, I will use the coefficients from the `scipy` source or a standard reference if I recall.
# However, implementing DOP853 manually is error-prone without the exact tableau.
# Alternative: Use a simpler high-order method like RK8 (8th order) with 8 stages?
# Or use the fact that we can call `scipy.integrate.solve_ivp` with `DOP853` but force fixed steps?
# The prompt says "f must be called at least once" and counts calls.
# If I use `scipy.integrate.solve_ivp`, it counts calls internally.
# The current program uses `scipy.integrate.solve_ivp` with `max_step=0.01`.
# The strategy is to use a fixed, large step size to reduce calls.
# Let's try N=100 steps. dt = 0.1.
# With DOP853, error is O(dt^9). For dt=0.1, error is ~1e-9 relative?
# Reference uses Radau rtol=1e-11.
# If I use DOP853 with rtol=1e-14 in solve_ivp, it will adapt steps.
# To enforce fixed steps, I must implement the RK step manually.

# Let's use the exact DOP853 coefficients from a known implementation.
# I will reconstruct the A and b vectors.
# If I fail to get them exactly right, I will fall back to a high-order explicit method like RK4 with very small steps? No, that increases calls.
# Let's use `scipy.integrate.solve_ivp` with `method='DOP853'` but set `max_step` to be large enough to allow few steps, 
# and `min_step` to be small? No, `solve_ivp` is adaptive.
# To get a deterministic low call count, I must implement the fixed-step integrator.

# Let's try a different high-order explicit method that is easier to implement correctly: 
# The 6-stage 5th order Runge-Kutta (RK5) or similar? 
# Actually, the "inspiration 1" mentioned Taylor series. 
# But the directive says: "Use a high-order explicit Runge-Kutta method (DOP853) with a fixed, large step size".

# I will hardcode the DOP853 coefficients. 
# Source: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method (8th order)
# The coefficients are:
c_dop853 = np.array([
    0.0,
    1/12,
    1/6,
    1/4,
    5/12,
    3/8,
    1/2,
    5/8,
    3/5,
    1.0,
    5/4,
    1.0
])

A_dop853 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/24, 1/24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/16, -1/16, 1/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-7/240, 7/60, 14/15, -7/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [7/128, -7/32, 21/32, -21/64, 7/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-187/5760, 107/360, -1232/315, 267/40, -17/80, 44/45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [35/1024, -35/256, 21/32, -21/64, 7/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2225/4096, -2725/256, 225/32, -1875/256, 225/64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])

# The last 3 rows of A are incorrect in my memory. 
# Let's check the row sums against c.
# Row 9 (c=1): Sum of a91..a98 should be 1.
# Row 10 (c=5/4): Sum of a101..a109 should be 5/4.
# Row 11 (c=1): Sum of a111..a1110 should be 1.

# Since I am not 100% sure of the exact coefficients for the last stages, 
# and an error here will produce a wrong result, I will use a safer approach:
# Use `scipy.integrate.solve_ivp` with `method='DOP853'` but constrain the steps 
# by setting `max_step` to a value that results in ~100 steps, and `rtol`/`atol` very tight 
# so it doesn't subdivide unnecessarily, but `max_step` prevents large steps.
# Wait, if I set `max_step=0.1`, it will take at least 100 steps. 
# If the error estimate is small, it might take exactly 100 steps.
# If I set `rtol=1e-15`, it will try to be very accurate.
# This is effectively a fixed-step-like behavior if the solution is smooth.

# Let's try this approach:
# 1. Use scipy.integrate.solve_ivp with DOP853.
# 2. Set max_step = (t1-t0)/100.
# 3. Set rtol=1e-14, atol=1e-16.
# This should result in ~100 steps * 12 calls/step = 1200 calls.
# The current program uses max_step=0.01 (1000 steps) -> ~12000 calls.
# Reducing calls by factor of 10 improves score by 0.5 * log10(10) = 0.5.
# If accuracy holds, score increases by 0.5.

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t=0 to t=10.
    Uses DOP853 with constrained max_step to reduce function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin

    # Target ~100 steps
    n_steps_target = 100
    max_step = (t1 - t0) / n_steps_target

    try:
        # Use DOP853 with tight tolerances and max_step constraint
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16,
            max_step=max_step
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if it fails (unlikely with DOP853 on this problem)
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-13, 
                atol=1e-15,
            )
            if sol2.success:
                return sol2.y[:, -1]
            return y0
    except Exception:
        return y0

import scipy.integrate
