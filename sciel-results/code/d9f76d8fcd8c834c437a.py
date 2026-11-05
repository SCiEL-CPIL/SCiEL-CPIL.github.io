"""ODE solver using spectral decomposition for linear systems, with fallback."""
import numpy as np
from scipy.linalg import eig, inv, expm


def _build_linear_model(f, t0, t1, n):
    """
    Attempts to determine A and b for f(t, y) = A y + b.
    Returns (A, b) if successful, else (None, None).
    """
    # Check time invariance
    y_test = np.random.RandomState(0).randn(n)
    y_test = y_test / np.linalg.norm(y_test)
    ft0 = f(t0, y_test)
    ft1 = f(t1, y_test)
    if not np.allclose(ft0, ft1, rtol=1e-9, atol=1e-12):
        return None, None

    # Get b
    b = f(t0, np.zeros(n))

    # Get A columns
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e) - b

    # Verify linearity on random vectors
    rng = np.random.RandomState(42)
    for _ in range(3):
        y_test = rng.randn(n)
        y_test = y_test / np.linalg.norm(y_test)
        f_val = f(t0, y_test)
        model_val = A @ y_test + b
        if not np.allclose(f_val, model_val, rtol=1e-9, atol=1e-12):
            return None, None

    return A, b


def _integrate_linear_eig(f, t_span, y0, A, b):
    """
    Integrate y' = A y + b using eigendecomposition.
    """
    t0, t1 = t_span
    dt = t1 - t0
    n = len(y0)

    # Handle homogeneous case
    if np.allclose(b, 0, atol=1e-14):
        try:
            w, V = eig(A)
            exp_w = np.exp(w * dt)
            c = np.linalg.solve(V, y0)
            y = V @ (exp_w * c)
        except np.linalg.LinAlgError:
            y = expm(A * dt) @ y0
        return y

    else:
        # Affine case
        # y(t) = e^{A dt} y0 + A^{-1} (e^{A dt} - I) b
        try:
            w, V = eig(A)

            # Compute term for y0
            c_y = np.linalg.solve(V, y0)
            y_part = V @ (np.exp(w * dt) * c_y)

            # Compute term for b
            term = np.zeros_like(w)
            mask = np.abs(w) > 1e-12
            term[mask] = (np.exp(w[mask] * dt) - 1) / w[mask]
            term[~mask] = dt

            c_b = np.linalg.solve(V, b)
            b_part = V @ (term * c_b)

            return y_part + b_part
        except np.linalg.LinAlgError:
            # Fallback to expm if eig fails
            n1 = n + 1
            M_aug = np.zeros((n1, n1))
            M_aug[:n, :n] = A
            M_aug[:n, n] = b
            z0 = np.zeros(n1)
            z0[:n] = y0
            z0[n] = 1.0
            M = expm(M_aug * dt)
            z = M @ z0
            return z[:n]


def _rk45_adaptive(f, t_span, y0, rtol=1e-12, atol=1e-14, max_steps=2000000):
    """Fallback: adaptive RK45 (Dormand-Prince) for nonlinear/time-dependent systems."""
    t0, t1 = t_span
    n = len(y0)
    y = np.array(y0, dtype=float)
    t = t0
    
    # Dormand-Prince coefficients
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    h = (t1 - t0) / 1000
    h = max(h, 1e-12)
    
    for step in range(max_steps):
        if t >= t1:
            break
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = [None] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            yi = y.copy()
            ti = t
            for j in range(i):
                yi = yi + h * A[i][j] * k[j]
                ti = ti + h * c[j]
            k[i] = f(ti, yi)
        
        # 5th order solution
        y_new = y.copy()
        for i in range(6):
            y_new = y_new + h * b[i] * k[i]
        
        # 4th order estimate for error
        y_est = y.copy()
        for i in range(6):
            y_est = y_est + h * b_star[i] * k[i]
        
        err = np.max(np.abs(y_new - y_est) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err <= 1.0:
            y = y_new
            t = t + h
            # Increase step
            if err > 0:
                h = h * min(5.0, max(0.2, 0.9 * err ** (-0.2)))
            else:
                h = h * 5.0
        else:
            # Reject step, decrease
            h = h * max(0.1, 0.9 * err ** (-0.25))
        
        if not np.all(np.isfinite(y)):
            return None
    
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try spectral decomposition approach for linear time-invariant systems
    try:
        A, b = _build_linear_model(f, t0, t1, n)
        if A is not None:
            y = _integrate_linear_eig(f, t_span, y0, A, b)
            if np.all(np.isfinite(y)):
                return y
    except Exception:
        pass
    
    # Fallback: adaptive RK45
    y = _rk45_adaptive(f, t_span, y0)
    if y is not None:
        return y
    
    # Final fallback: fixed-step RK4
    h = (t1 - t0) / 10000
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
