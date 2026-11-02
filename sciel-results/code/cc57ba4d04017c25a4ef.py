import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 120
SAFETY_MARGIN = 8

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv(x):
    """Linear autoconvolution via FFT, real part."""
    n = 2 * N
    fx = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(fx * fx.conj(), n=n)
    return ac[:N]  # lags 0..N-1

def _smooth_max_abs_ac(x, beta):
    """Log-sum-exp surrogate for max |autocorr|."""
    ac = _autoconv(x)
    a = np.abs(ac)
    m = a.max()
    # numerically stable logsumexp
    s = np.log(np.sum(np.exp(beta * (a - m)))) / beta + m
    return s

def _smooth_grad(x, beta):
    """Gradient of smooth max |autocorr| w.r.t. x."""
    ac = _autoconv(x)
    a = np.abs(ac)
    signs = np.sign(ac)
    m = a.max()
    w = np.exp(beta * (a - m))
    w = w / w.sum()
    # d|ac_k|/dx = signs[k] * d(ac_k)/dx
    # ac_k = sum_i x_i x_{i+k} (for k>=0), so d(ac_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
    # Vectorized: for each lag k, grad contribution is signs[k]*w[k]*(shift(x,k)+shift(x,-k))
    grad = np.zeros(N)
    for k in range(N):
        if k == 0:
            g = 2.0 * x
        else:
            g = np.zeros(N)
            g[:N-k] += x[k:]
            g[k:] += x[:N-k]
        grad += signs[k] * w[k] * g
    return grad

def _objective(x):
    ac = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _lbfgs_smooth(x0, beta, max_iter=2000, tol=1e-9):
    """Simple L-BFGS on smooth surrogate."""
    x = x0.copy()
    m = 10
    s_hist = []
    y_hist = []
    g = _smooth_grad(x, beta)
    for it in range(max_iter):
        if _check_deadline():
            break
        if np.linalg.norm(g) < tol:
            break
        # Two-loop recursion
        rho = [1.0 / (np.dot(y, s)) for s, y in zip(s_hist, y_hist)]
        q = g.copy()
        alpha = []
        for i in range(len(s_hist) - 1, -1, -1):
            a = rho[i] * np.dot(s_hist[i], q)
            alpha.append(a)
            q -= a * y_hist[i]
        r = (s_hist[-1] / np.dot(y_hist[-1], s_hist[-1])) * q if s_hist else q
        for i in range(len(s_hist)):
            b = rho[i] * np.dot(y_hist[i], r)
            r += (alpha[i] - b) * s_hist[i]
        d = -r
        # Line search (backtracking)
        f0 = _smooth_max_abs_ac(x, beta)
        gtd = np.dot(g, d)
        if gtd >= 0:
            d = -g
            gtd = np.dot(g, d)
        step = 1.0
        x_new = x + step * d
        f_new = _smooth_max_abs_ac(x_new, beta)
        for _ in range(30):
            if f_new <= f0 + 1e-4 * step * gtd:
                break
            step *= 0.5
            x_new = x + step * d
            f_new = _smooth_max_abs_ac(x_new, beta)
        s_new = x_new - x
        g_new = _smooth_grad(x_new, beta)
        y_new = g_new - g
        sy = np.dot(s_new, y_new)
        if sy > 1e-10:
            s_hist.append(s_new)
            y_hist.append(y_new)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        x = x_new
        g = g_new
    return x

def _coordinate_polish(x, num_iters=200):
    """Polish by adjusting individual elements to reduce top autocorr peaks."""
    x = x.copy()
    best_val = _objective(x)
    for it in range(num_iters):
        if _check_deadline():
            break
        ac = _autoconv(x)
        a = np.abs(ac)
        # Find top few peaks
        top_k = np.argsort(a)[-5:]
        improved = False
        for k in top_k:
            if a[k] < 1e-6:
                continue
            # Try flipping or adjusting each x_j to reduce |ac_k|
            # ac_k = sum_i x_i x_{i+k}; derivative w.r.t x_j is x_{j+k}+x_{j-k}
            # Try small perturbations
            for j in range(0, N, 2):
                old = x[j]
                # Try moving x[j] by small amount
                for delta in [0.5, -0.5, 1.0, -1.0]:
                    x[j] = old + delta
                    val = _objective(x)
                    if val < best_val:
                        best_val = val
                        improved = True
                        break
                    x[j] = old
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return x

def construct():
    """
    Returns the optimal sequence for the signed autocorrelation task.
    Uses multi-start L-BFGS on log-sum-exp surrogate + coordinate polish.
    """
    if _check_deadline():
        return np.ones(N)
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Multi-start: random, binary-like, low-freq
    starts = []
    starts.append(np.ones(N))
    for _ in range(3):
        starts.append(rng.standard_normal(N))
    # Binary-like: random +/-1
    for _ in range(2):
        starts.append(rng.choice([-1.0, 1.0], N))
    # Low-frequency structured
    for _ in range(2):
        freq = rng.uniform(0.5, 3.0)
        starts.append(np.sin(2 * np.pi * freq * np.arange(N) / N))
    
    beta = 20.0
    for x0 in starts:
        if _check_deadline():
            break
        # Normalize to have positive sum
        if np.sum(x0) < 0:
            x0 = -x0
        x_opt = _lbfgs_smooth(x0, beta, max_iter=1000, tol=1e-8)
        x_polished = _coordinate_polish(x_opt, num_iters=100)
        val = _objective(x_polished)
        if val < best_val:
            best_val = val
            best_x = x_polished
    
    # Final polish with more iterations if time
    if not _check_deadline():
        best_x = _coordinate_polish(best_x, num_iters=300)
    
    return best_x
