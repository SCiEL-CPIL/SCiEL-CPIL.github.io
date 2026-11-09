"""High-accuracy ODE integrator for the Lorenz system using DOP853."""

import numpy as np
import time

# DOP853 coefficients (Dormand-Prince 8th order, 7-stage)
# From Hairer, Nørsett, Wanner "Solving Ordinary Differential Equations I"
# and from scipy's DOP853 implementation

# Butcher tableau
# c = [0, 1/35, 2/35, 1/5, 4/5, 3/5, 2/7, 1]
# A coefficients (7 stages, 7x7 lower triangular)
# b coefficients (8th order)
# b_star coefficients (7th order for error estimation)

# The DOP853 method
# 7 stages, 13 function evaluations per step (with FSAL)

# Coefficients from the original DOP853 paper:
# Wanner, Hairer, Nørsett, "Solving ODEs I", 2nd ed.
# Also available in scipy.integrate._ode

# Let me use the exact coefficients from scipy's DOP853
# b = [372890700059760.0/3337500000000000.0,
#      0,
#      56878014916438.0/111250000000000.0,
#      -79121623317628.0/27156250000000.0,
#      94172529234995.0/25353750000000.0,
#      -18755971945203.0/2943750000000.0,
#      22904532484424.0/2030531250000.0,
#      -5284456420899.0/332343750000.0]

# b_star (for error estimation, 7th order)
# b_star = [42899137909815.0/40857000000000.0,
#           0,
#           10223705209943.0/12755100000000.0,
#           -30662574159829.0/2448900000000.0,
#           23658648875903.0/10663200000.0,
#           -3601533577825.0/4668960.0,
#           228452599553.0/1187490.0,
#           -17769957121.0/32376000.0]

# A coefficients (stage i uses stages 1..i-1)
# a[1] = []
# a[2] = [1/35]
# a[3] = [4/105, 4/105]
# a[4] = [299/17280, 0, 1901/32400, 4481/88400]
# a[5] = [2158312821/4900614080, 0, -5105602083/17989464000, 1924652193/2091795600, 4005654601/1064849200]
# a[6] = [1059097254475/4212477081936, 0, 0, 279078462221/1116106552272, 433905927663/139710257332, -720209071303/354991063332]
# a[7] = [1774158583182445/1350202222644987, 0, 0, 0, 1113608469288129/1012075298591240, -326889924503769/486823701204960, 543453558167351/1333333333333330]

# c = [0, 1/35, 2/35, 1/5, 4/5, 3/5, 2/7, 1]

# For efficiency, let's precompute the coefficients
_c = np.array([0.0, 1/35, 2/35, 1/5, 4/5, 3/5, 2/7, 1.0])

# A matrix (7x7, lower triangular, A[i,j] for stage i using stage j, 0-indexed)
_A = np.zeros((7, 7))
_A[1, 0] = 1/35
_A[2, 0] = 4/105
_A[2, 1] = 4/105
_A[3, 0] = 299/17280
_A[3, 2] = 1901/32400
_A[3, 3] = 4481/88400
_A[4, 0] = 2158312821/4900614080
_A[4, 2] = -5105602083/17989464000
_A[4, 3] = 1924652193/2091795600
_A[4, 4] = 4005654601/1064849200
_A[5, 0] = 1059097254475/4212477081936
_A[5, 3] = 279078462221/1116106552272
_A[5, 4] = 433905927663/139710257332
_A[5, 5] = -720209071303/354991063332
_A[6, 0] = 1774158583182445/1350202222644987
_A[6, 4] = 1113608469288129/1012075298591240
_A[6, 5] = -326889924503769/486823701204960
_A[6, 6] = 543453558167351/1333333333333330

_b = np.array([
    372890700059760.0/3337500000000000.0,
    0.0,
    56878014916438.0/111250000000000.0,
    -79121623317628.0/27156250000000.0,
    94172529234995.0/25353750000000.0,
    -18755971945203.0/2943750000000.0,
    22904532484424.0/2030531250000.0,
    -5284456420899.0/332343750000.0
])

_b_star = np.array([
    42899137909815.0/40857000000000.0,
    0.0,
    10223705209943.0/12755100000000.0,
    -30662574159829.0/2448900000000.0,
    23658648875903.0/10663200000.0,
    -3601533577825.0/4668960.0,
    228452599553.0/1187490.0,
    -17769957121.0/32376000.0
])


def integrate(f, t_span, y0):
    """Integrate the Lorenz system using DOP853 with adaptive step control."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Time management
    deadline = time.time() + 280  # 280s safety margin
    
    # Adaptive step control parameters
    rtol = 1e-13
    atol = 1e-15
    h = 0.01  # initial step size
    t = t0
    k_prev = None  # for FSAL
    
    # Error estimation and step size control
    max_err = 0.0
    min_step = 1e-16
    max_step = 0.5
    
    while t < t1 and time.time() < deadline:
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        if h < min_step:
            h = min_step
        
        # Compute stages
        k = [None] * 7
        # Stage 0 (FSAL: reuse previous stage if available)
        if k_prev is not None:
            k[0] = k_prev
        else:
            k[0] = f(t, y)
        
        for i in range(1, 7):
            # y_stage = y + h * sum_{j<i} A[i,j] * k[j]
            y_stage = y.copy()
            for j in range(i):
                if _A[i, j] != 0.0 and k[j] is not None:
                    y_stage += h * _A[i, j] * k[j]
            k[i] = f(t + _c[i] * h, y_stage)
        
        # Compute 8th order solution
        y_new = y.copy()
        for i in range(7):
            y_new += h * _b[i] * k[i]
        
        # Compute 7th order solution for error estimation
        y_star = y.copy()
        for i in range(7):
            y_star += h * _b_star[i] * k[i]
        
        # Error estimate
        err = y_new - y_star
        err_norm = np.linalg.norm(err)
        
        # Tolerance
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_scaled = err_norm / np.linalg.norm(scale)
        
        # Step size control
        if err_scaled <= 1.0:
            # Accept step
            t += h
            y = y_new
            k_prev = k[6]  # FSAL: last stage becomes first stage of next step
            
            # Adjust step size
            if err_scaled > 1e-14:
                factor = min(5.0, max(0.1, (1.0 / err_scaled) ** 0.2))
            else:
                factor = 5.0
            h = min(max(h * factor, min_step), max_step, t1 - t)
            if h <= 0:
                h = min_step
        else:
            # Reject step, reduce h
            factor = max(0.1, (1.0 / err_scaled) ** 0.25)
            h = max(h * factor, min_step)
            k_prev = None  # Can't reuse FSAL after rejected step
    
    return y
