"""Rewritten program: signed autocorrelation maximum absolute-value minimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE_FRACTION = 0.85  # use 85% of the 150s budget to be safe


def _autoconv_abs_max(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Autoconvolution via FFT: conv(x, x)
    # Use real FFT for efficiency
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)
    # The autoconvolution has length 2*n-1, but irfft gives 2*n
    # We want the full convolution, which is ac[:2*n-1] but actually
    # for autocorrelation (not convolution), it's different.
    # Wait, the task says "autoconvolution" which is conv(x, x).
    # Let's use the full convolution result.
    return np.max(np.abs(ac[:2 * n - 1]))


def _squared_integral(x):
    """Compute (sum(x))^2, which is the squared integral for step functions."""
    s = np.sum(x)
    return s * s


def _score(x):
    """Compute the ratio: max |autoconv| / (sum(x))^2."""
    denom = _squared_integral(x)
    if denom < 1e-12:
        return np.inf
    return _autoconv_abs_max(x) / denom


def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try several initializations and optimization runs
    rng = np.random.RandomState(42)
    
    # Different initialization strategies
    inits = [
        np.ones(N),
        rng.choice([-1.0, 1.0], size=N),
        rng.randn(N),
        np.sign(np.sin(np.linspace(0, 4 * np.pi, N))),
        np.sign(np.cos(np.linspace(0, 2 * np.pi, N))),
        np.sign(np.sin(np.linspace(0, 8 * np.pi, N))),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        x0 = init.astype(np.float64)
        
        # Objective function for optimization
        def objective(x):
            return _score(x)
        
        # Use L-BFGS-B for bounded optimization
        # We'll bound values to [-2, 2] to keep things reasonable
        bounds = [(-2.0, 2.0)] * N
        
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Post-processing: try to snap to {-1, 0, 1} or {-1, 1}
    # First, try rounding to nearest integer in {-1, 0, 1}
    candidates = [
        np.clip(best_x, -1, 1),
        np.sign(best_x),
        np.where(np.abs(best_x) > 0.5, np.sign(best_x), 0),
        np.round(best_x).clip(-1, 1),
    ]
    
    for cand in candidates:
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_x = cand.copy()
    
    # Local search: try flipping individual elements
    if time.time() < deadline:
        x = best_x.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                orig = x[i]
                # Try flipping sign
                x[i] = -orig
                s = _score(x)
                if s < best_score - 1e-12:
                    best_score = s
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = orig
                # Try setting to 0
                if x[i] != 0:
                    x[i] = 0
                    s = _score(x)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_x = x.copy()
                        improved = True
                    else:
                        x[i] = orig
    
    return best_x


if __name__ == '__main__':
    result = construct()
    print("Score:", _score(result))
