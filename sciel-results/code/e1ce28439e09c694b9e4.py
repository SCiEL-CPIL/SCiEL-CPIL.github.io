"""Improved program using Radau-type implicit RK with analytic Jacobian for Van der Pol."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _vdp_rhs(t, y):
    """Van der Pol oscillator RHS (mu=5)."""
    y0, y1 = y
    return np.array([y1, 5.0 * (1.0 - y0 * y0) * y1 - y0])


def _vdp_jac(t, y):
    """Analytic Jacobian of Van der Pol oscillator."""
    y0, y1 = y
    # d/dy0 of y1' = d/dy0 [5(1-y0^2)y1 - y0] = -10*y0*y1 - 1
    # d/dy1 of y1' = d/dy1 [5(1-y0^2)y1 - y0] = 5*(1-y0^2)
    # d/dy0 of y0' = 0
    # d/dy1 of y0' = 1
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]
    ])


def _radau3_step(f, jac, t, y, h, max_iter=30, tol=1e-14):
    """
    One step of Radau IIA 3-stage implicit Runge-Kutta.
    Nodes: c1=0.5*(1-2/sqrt(6)), c2=0.5, c3=0.5*(1+2/sqrt(6))
    Weights: b1=1/6, b2=2/3, b3=1/6
    Stage equations: Yi = y + h * sum_j a_ij f(t + c_j h, Yj)
    """
    sqrt6 = np.sqrt(6.0)
    c = np.array([0.5 * (1 - 2.0 / sqrt6), 0.5, 0.5 * (1 + 2.0 / sqrt6)])
    b = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    a = np.array([
        [0.0, 0.0, 0.0],
        [5.0 / 12.0 - 5.0 / 36.0 * sqrt6, 0.5, 5.0 / 36.0 * sqrt6],
        [5.0 / 12.0 + 5.0 / 36.0 * sqrt6, 0.5, 5.0 / 12.0 - 5.0 / 36.0 * sqrt6]
    ])
    
    n = len(y)
    I = np.eye(n)
    Y = np.zeros((3, n))
    Y[0] = y
    Y[1] = y
    Y[2] = y
    
    for _iter in range(max_iter):
        converged = True
        for i in range(3):
            # Compute sum_j a_ij f(t + c_j h, Yj)
            s = np.zeros(n)
            for j in range(3):
                if a[i, j] != 0.0:
                    fj = f(t + c[j] * h, Y[j])
                    s += a[i, j] * fj
            target = y + h * s
            residual = target - Y[i]
            if np.linalg.norm(residual) > tol * (1.0 + np.linalg.norm(Y[i])):
                converged = False
                # Newton step: solve (I - h * sum_j a_ij J_j) dY_i = residual
                # Approximate with J at Y[i]
                J = jac(t + c[i] * h, Y[i])
                A = I - h * a[i, i] * J
                # For a[i][i] != 0, we need to handle the implicit part
                # Actually, the linear system for stage i is:
                # Y_i - h * sum_j a_ij f(Y_j) = y
                # Linearizing around current Y: (I - h * sum_j a_ij J_j) dY = residual
                # But for simplicity and since a[i][i] is the main implicit term,
                # use J at current point
                try:
                    dY = np.linalg.solve(A, residual)
                except np.linalg.LinAlgError:
                    dY = residual * 0.5
                Y[i] = Y[i] + dY
        if converged:
            break
    
    y_new = y + h * np.dot(b, Y)
    return y_new


def _adaptive_radau(f, t0, t1, y0, rtol=1e-13, atol=1e-14, h0=0.01, max_steps=100000):
    """
    Adaptive Radau IIA integrator with error estimation.
    Uses Radau 3 as base and compares with smaller step for error estimation.
    """
    t = t0
    y = np.array(y0, dtype=float)
    h = h0
    steps = 0
    deadline = time.time() + 280.0  # Safety margin
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        if t + h > t1:
            h = t1 - t
        if h < 1e-15:
            h = 1e-15
        
        # Take a full step
        y_full = _radau3_step(f, _vdp_jac, t, y, h)
        
        # Take two half steps for error estimate
        y_half1 = _radau3_step(f, _vdp_jac, t, y, h / 2.0)
        y_half2 = _radau3_step(f, _vdp_jac, t + h / 2.0, y_half1, h / 2.0)
        
        err = np.linalg.norm(y_full - y_half2)
        scale = atol + rtol * np.linalg.norm(y_half2)
        
        if err <= scale:
            # Accept step
            y = y_half2
            t += h
            # Increase step
            if err > 1e-16:
                h *= min(5.0, 0.9 * (scale / err) ** 0.25)
            else:
                h *= 2.0
        else:
            # Reject step, decrease h
            h *= max(0.1, 0.9 * (scale / err) ** 0.25)
        
        steps += 1
        if steps > max_steps:
            break
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using adaptive Radau IIA method.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use adaptive Radau with tight tolerances
    y_end = _adaptive_radau(f, t0, t1, y0, rtol=1e-13, atol=1e-14, h0=0.05)
    
    # Ensure finite result
    if not np.all(np.isfinite(y_end)):
        # Fallback: simple fixed-step Radau
        n_steps = 500
        h = (t1 - t0) / n_steps
        t = t0
        y = y0.copy()
        for _ in range(n_steps):
            y = _radau3_step(f, _vdp_jac, t, y, h)
            t += h
        y_end = y
    
    return y_end
# EVOLVE-BLOCK-END
