import numpy as np
import time

N = 400

def compute_score(f):
    """Compute max |autoconv| / (sum)^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    A = np.convolve(f, f, mode='full')
    max_abs = np.max(np.abs(A))
    return max_abs / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = np.inf
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    x = np.arange(N)
    center = N / 2
    
    # Try Gaussian windows with various widths
    for sigma in [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400]:
        if time.time() > deadline:
            break
        f = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        s = compute_score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Try triangular windows
    for width in [100, 150, 200, 250, 300]:
        if time.time() > deadline:
            break
        f = np.zeros(N)
        half = width // 2
        c = N // 2
        f[c-half:c+half+1] = np.concatenate([
            np.linspace(1, 0, half+1),
            np.linspace(0, 1, half)
        ])
        s = compute_score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Local optimization with gradient descent
    if time.time() < deadline - 20:
        f = best_f.copy()
        lr = 0.005
        eps = 1e-4
        for step in range(100):
            if time.time() > deadline:
                break
            grad = np.zeros(N)
            for i in range(N):
                f_plus = f.copy()
                f_plus[i] += eps
                f_minus = f.copy()
                f_minus[i] -= eps
                grad[i] = (compute_score(f_plus) - compute_score(f_minus)) / (2 * eps)
            f -= lr * grad
            s = compute_score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    return best_f
