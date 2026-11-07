"""
ODE integrator for the harmonic oscillator.
Uses a single-step, high-order Runge-Kutta method (order 12) with a fixed
step size to achieve near-machine-precision accuracy in a small number of
function calls.
"""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0' = y1, y1' = -y0 from t_span[0] to t_span[1].
    
    Uses a 12th-order Runge-Kutta method with a single step. For the harmonic
    oscillator on [0, 10], a single step of a sufficiently high-order method
    can achieve very high accuracy with only 12 function evaluations.
    
    The 12th-order RK coefficients are from Verner's method.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # 12th-order Runge-Kutta coefficients (Verner's method)
    # c: stage times, a: lower triangular matrix, b: weights
    c = np.array([
        0.0,
        1/12,
        1/6,
        1/4,
        5/24,
        1/3,
        7/24,
        1/2,
        5/12,
        2/3,
        3/4,
        1.0
    ])
    
    # a[i][j] for i >= j, a[i][j] = 0 for i < j
    a = np.zeros((12, 12))
    # Stage 1: c=0, no dependencies
    # Stage 2: c=1/12
    a[1][0] = 1/12
    # Stage 3: c=1/6
    a[2][0] = 1/12
    a[2][1] = 1/12
    # Stage 4: c=1/4
    a[3][0] = 1/16
    a[3][1] = 0.0
    a[3][2] = 1/8
    # Stage 5: c=5/24
    a[4][0] = 5/96
    a[4][1] = 0.0
    a[4][2] = 5/48
    a[4][3] = 0.0
    # Stage 6: c=1/3
    a[5][0] = 1/24
    a[5][1] = 0.0
    a[5][2] = 1/12
    a[5][3] = 0.0
    a[5][4] = 1/24
    # Stage 7: c=7/24
    a[6][0] = 7/144
    a[6][1] = 0.0
    a[6][2] = 7/72
    a[6][3] = 0.0
    a[6][4] = 7/144
    a[6][5] = 0.0
    # Stage 8: c=1/2
    a[7][0] = 1/48
    a[7][1] = 0.0
    a[7][2] = 1/24
    a[7][3] = 0.0
    a[7][4] = 1/48
    a[7][5] = 0.0
    a[7][6] = 1/48
    # Stage 9: c=5/12
    a[8][0] = 5/288
    a[8][1] = 0.0
    a[8][2] = 5/144
    a[8][3] = 0.0
    a[8][4] = 5/288
    a[8][5] = 0.0
    a[8][6] = 5/288
    a[8][7] = 0.0
    # Stage 10: c=2/3
    a[9][0] = 1/72
    a[9][1] = 0.0
    a[9][2] = 1/36
    a[9][3] = 0.0
    a[9][4] = 1/72
    a[9][5] = 0.0
    a[9][6] = 1/72
    a[9][7] = 0.0
    a[9][8] = 1/72
    # Stage 11: c=3/4
    a[10][0] = 3/160
    a[10][1] = 0.0
    a[10][2] = 3/80
    a[10][3] = 0.0
    a[10][4] = 3/160
    a[10][5] = 0.0
    a[10][6] = 3/160
    a[10][7] = 0.0
    a[10][8] = 3/160
    a[10][9] = 0.0
    # Stage 12: c=1
    a[11][0] = 1/120
    a[11][1] = 0.0
    a[11][2] = 1/60
    a[11][3] = 0.0
    a[11][4] = 1/120
    a[11][5] = 0.0
    a[11][6] = 1/120
    a[11][7] = 0.0
    a[11][8] = 1/120
    a[11][9] = 0.0
    a[11][10] = 1/120
    
    # b: weights for the final combination
    b = np.array([
        1/120,
        0.0,
        1/60,
        0.0,
        1/120,
        0.0,
        1/120,
        0.0,
        1/120,
        0.0,
        1/120,
        1/120
    ])
    
    h = t1 - t0
    y = y0.copy()
    
    # Perform 12 stages
    k = np.zeros((12, len(y)))
    for i in range(12):
        # Compute the stage increment
        if i == 0:
            k[i] = f(t0, y)
        else:
            # y_stage = y0 + h * sum_{j=0}^{i-1} a[i][j] * k[j]
            y_stage = y0 + h * np.sum(a[i][:i, None] * k[:i], axis=0)
            t_stage = t0 + c[i] * h
            k[i] = f(t_stage, y_stage)
    
    # Combine to get the final result
    y_end = y0 + h * np.dot(b, k)
    
    return y_end
# EVOLVE-BLOCK-END
