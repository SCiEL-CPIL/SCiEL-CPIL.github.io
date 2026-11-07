import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE_OFFSET = 145.0

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _smooth_objective(h, lam):
    """
    Smoothed objective: log-sum-exp of correlations.
    We want to minimize max(C_k).
    C_k = correlate(h, 1-h)[k].
    """
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    shifted = corr - m
    # log(sum(exp(lam * shifted))) / lam approximates max
    # To avoid overflow, we subtract max first.
    # Actually, log-sum-exp(x) = max(x) + log(sum(exp(x - max(x))))
    # We want log-sum-exp(lam * corr) / lam.
    # = (lam * m + log(sum(exp(lam * (corr - m))))) / lam
    # = m + (1/lam) * log(sum(exp(lam * (corr - m))))
    
    # If lam is large, this is close to m.
    # If lam is small, it's close to mean.
    # We want to start with small lam (smooth) and increase.
    
    # Note: corr can be negative?
    # h in [0,1], 1-h in [0,1]. corr is sum of products, so >= 0.
    
    val = m + np.log(np.sum(np.exp(lam * shifted))) / lam
    return val

def _optimize_sequence(h_start, time_budget=10.0):
    """
    Optimize a starting sequence using SLSQP with increasing smoothness.
    """
    start_time = time.time()
    current_h = h_start.copy()
    best_h = h_start.copy()
    best_val = _correlation_value(best_h)
    
    # Increasing lambda makes the surrogate closer to the true max
    # We use a geometric progression of lambda
    lambdas = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for lam in lambdas:
        if time.time() - start_time > time_budget - 2.0:
            break
        
        def obj(x):
            return _smooth_objective(x, lam)
        
        try:
            res = minimize(
                obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass
    
    # Final exact optimization
    if time.time() - start_time < time_budget:
        try:
            def exact_obj(x):
                corr = np.correlate(x, 1 - x, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Start 1: Uniform
    h_uniform = np.full(N, 0.5)
    
    # Start 2: Random Binary
    rng = np.random.default_rng(42)
    h_binary = rng.integers(0, 2, N).astype(float)
    h_binary = _project_to_feasible(h_binary)
    
    # Start 3: M-sequence (if we can generate one easily)
    # For N=100, m-sequences don't exist (length must be 2^k - 1).
    # So we use random binary with projection.
    
    starts = [h_uniform, h_binary]
    
    # Add some structured starts
    # Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Blocks
    for block_size in [4, 5, 10, 20]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
        
    # Sine waves
    for freq in [1, 2, 3, 5]:
        h_sin = 0.5 + 0.5 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))

    # Optimize each start
    # We have ~145s budget.
    # Each optimization takes ~5-10s.
    # We can do ~10-15 optimizations.
    
    num_starts = len(starts)
    time_per_start = (deadline - start_time - 5.0) / num_starts
    
    for idx, h_start in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        current_budget = min(time_per_start, deadline - start_time - 5.0)
        if current_budget < 2.0:
            break
            
        try:
            h_opt = _optimize_sequence(h_start, time_budget=current_budget)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final polish on the best found
    if time.time() < deadline - 10.0:
        try:
            best_h = _optimize_sequence(best_h, time_budget=10.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    return best_h
