"""Rewritten Robertson ODE solver using variable-step Radau IIA (3-stage, order 3)."""
# EVOLVE-BLOCK-START
import numpy as np


# Radau IIA coefficients for s=3 (order 3)
# B matrix (3x3) and C vector
_RADAU_B = np.array([
    [0.19815289934589786, 0.0, 0.0],
    [0.12976165850964382, 0.19815289934589786, 0.0],
    [0.04150568823649234, 0.12976165850964382, 0.19815289934589786],
])
_RADAU_C = np.array([
    0.19815289934589786,
    0.3963057986917957,
    1.0,
])
# b vector for the final update
_RADAU_B_VEC = np.array([
    0.04150568823649234,
    0.12976165850964382,
    0.19815289934589786,
])
# bhat for error estimation (order 2)
_RADAU_BHAT = np.array([
    0.02186411742315574,
    0.09116166610279457,
    0.17109479999350958,
])


def _robertson_jacobian(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    # y' = [-y0*y1, y0*y1 - 1e4*y1*y2, 1e4*y1*y2]
    # d/dy0: [-y1, y1, 0]
    # d/dy1: [-y0, -y0 - 1e4*y2, 1e4*y2]
    # d/dy2: [0, -1e4*y1, 1e4*y1]
    J = np.array([
        [-y1, y1, 0.0],
        [-y0, -y0 - 1e4 * y2, 1e4 * y2],
        [0.0, -1e4 * y1, 1e4 * y1],
    ])
    return J


def _robertson_rhs(t, y):
    """Robertson problem RHS."""
    y0, y1, y2 = y
    return np.array([
        -y0 * y1,
        y0 * y1 - 1e4 * y1 * y2,
        1e4 * y1 * y2,
    ])


def _solve_nonlinear_system(f, t, y, h, J_base, max_newton=8):
    """
    Solve the nonlinear system for Radau IIA stages.
    y_i = y0 + h * sum_j B_ij * f(t + C_j*h, y_j)
    
    We use Newton's method with the analytic Jacobian.
    """
    n = len(y)
    s = len(_RADAU_C)
    
    # Initial guess: y_i = y (backward Euler-like guess)
    Y = np.zeros((s, n))
    for i in range(s):
        Y[i] = y.copy()
    
    for newton_iter in range(max_newton):
        # Evaluate f at all stages
        F = np.zeros((s, n))
        for i in range(s):
            F[i] = f(t + _RADAU_C[i] * h, Y[i])
        
        # Compute residual: R_i = Y_i - y - h * sum_j B_ij * F_j
        R = np.zeros((s, n))
        for i in range(s):
            R[i] = Y[i] - y - h * (_RADAU_B[i] @ F)
        
        # Check convergence
        max_res = np.max(np.abs(R))
        if max_res < 1e-12 * (1.0 + np.max(np.abs(Y))):
            break
        
        # Build the big Jacobian for Newton step
        # The system is: Y_i - h * sum_j B_ij * f(Y_j) = y
        # d/dY_k: delta_ik I - h * B_ik * J_k
        # where J_k = df/dy at stage k
        
        # For efficiency, we solve the coupled system.
        # The Jacobian of the residual with respect to Y is a block matrix:
        # M[i,k] = delta_ik * I - h * B[i,k] * J_k
        # We need to solve M * delta_Y = -R
        
        # Since s=3 and n=3, we can build the full 9x9 matrix
        dim = s * n
        M = np.zeros((dim, dim))
        RHS = np.zeros(dim)
        
        for i in range(s):
            for k in range(s):
                block_start_i = i * n
                block_start_k = k * n
                if i == k:
                    # I - h * B[i,i] * J_i
                    J_i = _robertson_jacobian(t + _RADAU_C[i] * h, Y[i])
                    M[block_start_i:block_start_i+n, block_start_k:block_start_k+n] = \
                        np.eye(n) - h * _RADAU_B[i, i] * J_i
                else:
                    J_k = _robertson_jacobian(t + _RADAU_C[k] * h, Y[k])
                    M[block_start_i:block_start_i+n, block_start_k:block_start_k+n] = \
                        -h * _RADAU_B[i, k] * J_k
                RHS[block_start_i:block_start_i+n] = -R[i]
        
        # Solve
        try:
            delta_Y = np.linalg.solve(M, RHS)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            delta_Y = np.linalg.lstsq(M, RHS, rcond=None)[0]
        
        # Update
        for i in range(s):
            Y[i] += delta_Y[i*n:(i+1)*n]
    
    return Y, F


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using variable-step Radau IIA.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size
    h = (t1 - t0) * 1e-3
    h_min = (t1 - t0) * 1e-10
    h_max = (t1 - t0) * 0.1
    
    # Error tolerances
    rtol = 1e-10
    atol = 1e-12
    
    while t < t1:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        if h < h_min:
            break
        
        # Solve the Radau IIA stages
        Y, F = _solve_nonlinear_system(f, t, y, h, None)
        
        # New y
        y_new = y + h * (_RADAU_B_VEC @ F)
        
        # Error estimate
        y_err = h * ((_RADAU_B_VEC - _RADAU_BHAT) @ F)
        
        # Error norm
        err_norm = np.max(np.abs(y_err) / (atol + rtol * np.abs(y_new)))
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t = t + h
            
            # Adapt step size
            if err_norm < 1e-6:
                h *= 5.0
            elif err_norm < 0.5:
                h *= 3.0
            elif err_norm < 0.1:
                h *= 1.5
            else:
                h *= 1.0
        else:
            # Reject step
            h *= max(0.1, 0.9 * err_norm ** (-0.2))
        
        h = min(h, h_max)
        h = max(h, h_min)
    
    return y
# EVOLVE-BLOCK-END
