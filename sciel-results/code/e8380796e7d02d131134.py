# EVOLVE-BLOCK-START
import numpy as np
import time

# Radau IIA(5) coefficients
_RADAU_NODES = np.array([
    0.016583123118716889,
    0.10673702345167850,
    0.29859127626016403,
    0.56153892752212236,
    0.93155055054732002
])

_RADAU_BETA = _RADAU_NODES.copy()

def _compute_radau_ii_a(nodes):
    """Compute a[i][j] = integral_0^{c_i} l_j(t) dt for Radau IIA."""
    n = len(nodes)
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            poly = np.array([1.0])
            for k in range(n):
                if k != j:
                    poly = np.convolve(poly, np.array([1.0, -nodes[k]]))
            denom = np.prod([nodes[j] - nodes[k] for k in range(n) if k != j])
            poly /= denom
            integral = 0.0
            for k in range(len(poly)):
                power = len(poly) - 1 - k
                integral += poly[k] * (nodes[i] ** (power + 1)) / (power + 1)
            a[i, j] = integral
    return a

_A = _compute_radau_ii_a(_RADAU_NODES)


def _vdp_jacobian(y):
    """Analytic Jacobian for Van der Pol oscillator (mu=5)."""
    y0 = y[0]
    J = np.array([
        [0.0, 1.0],
        [-(1.0 + 10.0 * y0 * y0), 5.0 * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    deadline = time.time() + 280.0
    
    # Use adaptive step size with Radau IIA(5)
    h = (t1 - t0) / 50.0
    t = t0
    
    rtol = 1e-12
    atol = 1e-13
    SAFETY = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 5.0
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        
        if t + h > t1:
            h = t1 - t
        
        # Radau IIA(5) step with Newton iteration
        Y = np.zeros((n, n))
        for i in range(n):
            Y[:, i] = y + _RADAU_NODES[i] * h * f(t, y)
        
        converged = False
        for newton_iter in range(20):
            F = np.zeros((n, n))
            for i in range(n):
                F[:, i] = f(t + _RADAU_NODES[i] * h, Y[:, i])
            
            R = np.zeros((n, n))
            for i in range(n):
                R[:, i] = Y[:, i] - y - h * (_A[i] @ F)
            
            R_norm = np.max(np.abs(R))
            if R_norm < atol * 0.01:
                converged = True
                break
            
            # Build block Jacobian
            N_total = n * n
            J_full = np.eye(N_total)
            for i in range(n):
                Ji = _vdp_jacobian(Y[:, i])
                for j in range(n):
                    rs, cs = i * n, j * n
                    if i == j:
                        J_full[rs:rs+n, cs:cs+n] = np.eye(n) - h * Ji
                    else:
                        J_full[rs:rs+n, cs:cs+n] = -h * _A[i, j] * Ji
            
            R_flat = R.flatten()
            try:
                delta = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                delta, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
            
            delta = delta.reshape((n, n))
            Y += delta
            
            if np.linalg.norm(delta) < 1e-15 * (1.0 + np.linalg.norm(Y)):
                converged = True
                break
        
        if not converged:
            h *= 0.5
            if h < 1e-12:
                break
            continue
        
        # Compute new y
        F = np.zeros((n, n))
        for i in range(n):
            F[:, i] = f(t + _RADAU_NODES[i] * h, Y[:, i])
        y_new = y + h * (_RADAU_BETA @ F)
        
        # Error estimate using residual magnitude
        err = np.max(np.abs(y_new - y))
        scale = atol + rtol * np.max(np.abs(y_new))
        err_ratio = err / scale if scale > 0 else err
        
        if err_ratio > 1.0:
            factor = SAFETY * err_ratio ** (-0.2)
            factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
            h *= factor
            continue
        else:
            y = y_new
            t += h
            if err_ratio < 1e-4:
                factor = SAFETY * err_ratio ** (-0.2)
                factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
                h *= factor
    
    return y
# EVOLVE-BLOCK-END
