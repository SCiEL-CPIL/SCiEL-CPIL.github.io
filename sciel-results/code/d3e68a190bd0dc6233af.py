"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorrelation(x):
    """Compute full autocorrelation of sequence x (length N)."""
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.fft(x, size)
    acf = np.fft.ifft(X * np.conj(X)).real
    return acf[:n]  # lags 0 to n-1

def _compute_c3(x):
    """Compute the C3 upper bound: max absolute sidelobe."""
    acf = _autocorrelation(x)
    sidelobes = acf[1:]
    return np.max(np.abs(sidelobes))

def _lse_surrogate(x, beta=50.0):
    """Smooth surrogate: log-sum-exp of squared sidelobes."""
    acf = _autocorrelation(x)
    sidelobes_sq = acf[1:] ** 2
    m = np.max(sidelobes_sq)
    if m < 1e-30:
        return m
    return m + np.log(np.sum(np.exp((sidelobes_sq - m) * beta)) / N) / beta

def _lse_grad(x, beta=50.0):
    """Gradient of log-sum-exp surrogate w.r.t. x."""
    acf = _autocorrelation(x)
    sidelobes = acf[1:]
    sidelobes_sq = sidelobes ** 2
    m = np.max(sidelobes_sq)
    weights = np.exp((sidelobes_sq - m) * beta)
    weights = weights / np.sum(weights)
    # d/dx_k of acf[j] for lag j: sum over i of x[i]*x[i+j] (for j>=0)
    # Gradient contribution: 2 * sum_j w_j * d(acf[j])/dx
    # d(acf[j])/dx_k = x[k-j] + x[k+j] (where in range) for the linear part
    # Actually for autocorrelation: acf[j] = sum_i x[i]*x[i+j]
    # d(acf[j])/dx_k = x[k-j] + x[k+j] if both in range
    grad = np.zeros(N)
    for j in range(1, N):
        wj = weights[j - 1] * 2.0
        # d(acf[j])/dx_k = x[k-j] + x[k+j] where valid
        # For k in [0, N-1]:
        #   x[k-j] valid when k >= j
        #   x[k+j] valid when k+j < N
        if j < N:
            # Contribution from x[k-j]: k from j to N-1
            grad[j:] += wj * x[:N - j]
            # Contribution from x[k+j]: k from 0 to N-1-j
            grad[:N - j] += wj * x[j:]
    return grad

def _continuous_relaxation(rng, time_limit=60.0):
    """Projected gradient descent on continuous relaxation with LSE surrogate."""
    start_time = time.time()
    # Initialize with random ±1
    x = rng.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_val = _lse_surrogate(best_x)
    
    # Learning rate schedule
    lr = 0.1
    beta = 50.0
    
    for iteration in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        grad = _lse_grad(x, beta)
        
        # Gradient step
        x = x - lr * grad
        
        # Projection: threshold to nearest ±1 (hard projection)
        x_proj = np.sign(x)
        x_proj[x_proj == 0] = 1.0
        
        # Evaluate
        val = _lse_surrogate(x_proj)
        if val < best_val - 1e-10:
            best_val = val
            best_x = x_proj.copy()
        
        # Decay learning rate
        lr *= 0.999
        beta *= 1.001  # Increase sharpness over time
        
        # Periodic restarts to escape local minima
        if iteration % 500 == 0 and iteration > 0:
            x = rng.choice([-1.0, 1.0], size=N)
    
    return best_x, _compute_c3(best_x)

def _local_search(x, rng, time_limit=10.0):
    """Fast local search: flip bits to reduce C3."""
    best_x = x.copy()
    best_c3 = _compute_c3(best_x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for pos in range(N):
            x_trial = best_x.copy()
            x_trial[pos] *= -1.0
            c3_trial = _compute_c3(x_trial)
            if c3_trial < best_c3 - 1e-10:
                best_c3 = c3_trial
                best_x = x_trial
                improved = True
                break
    
    return best_x, best_c3

def construct():
    rng = np.random.default_rng(42)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s (evaluator kills at 150s)
    
    # Phase 1: Continuous relaxation via projected gradient descent
    if time.time() < deadline - 20:
        x, c3 = _continuous_relaxation(rng, time_limit=min(80.0, deadline - time.time() - 20))
    else:
        x = rng.choice([-1.0, 1.0], size=N)
        c3 = _compute_c3(x)
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 5:
        x, c3 = _local_search(x, rng, time_limit=min(15.0, deadline - time.time() - 5))
    
    # Ensure we return a valid ±1 sequence
    x = np.sign(x)
    x[x == 0] = 1
    
    return x
# EVOLVE-BLOCK-END
