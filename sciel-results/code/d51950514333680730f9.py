"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    # h is a numpy array of size N
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initial best is the seed (all 0.5)
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Helper to check validity and update best
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        # Enforce bounds [0,1] and sum constraint approximately
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        # Adjust sum to be exactly N/2 if slightly off
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
        val = _objective(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()

    # 1. Try structured initializations
    np.random.seed(42)
    
    # List of initializations to try
    inits = [
        best_h.copy(),
        np.linspace(0.0, 1.0, N, endpoint=False),
        np.linspace(1.0, 0.0, N, endpoint=False),
        np.random.uniform(0, 1, N),
        np.where(np.arange(N) % 2 == 0, 0.8, 0.2),
        np.where(np.arange(N) % 2 == 1, 0.8, 0.2),
        # Periodic 5
        np.tile([0.0, 0.0, 1.0, 1.0, 0.5], 20),
        # Periodic 4
        np.tile([0.0, 0.5, 1.0, 0.5], 25),
    ]
    
    # Add sinusoidal inits from inspiration
    def make_sinusoidal_init(frequencies, amplitudes, phase=0.0):
        h = np.zeros(N)
        for f, a in zip(frequencies, amplitudes):
            h += a * np.cos(2 * np.pi * f * np.arange(N) / N + phase)
        h = 0.5 + h
        h = np.clip(h, 0, 1)
        h = h + (N / 2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        return h

    inits.append(make_sinusoidal_init([1], [0.4]))
    inits.append(make_sinusoidal_init([2], [0.4]))
    inits.append(make_sinusoidal_init([3], [0.4]))
    inits.append(make_sinusoidal_init([1, 3], [0.3, 0.2]))
    inits.append(make_sinusoidal_init([1, 2, 4], [0.25, 0.2, 0.15]))
    inits.append(make_sinusoidal_init([2, 3, 5], [0.25, 0.2, 0.15]))
    
    # Block patterns from inspiration 2
    h_block = np.zeros(N)
    h_block[:N // 2] = 1.0
    h_block = h_block + (N / 2.0 - np.sum(h_block)) / N
    h_block = np.clip(h_block, 0, 1)
    inits.append(h_block)
    
    h_three = np.zeros(N)
    h_three[:N // 3] = 1.0
    h_three[2 * N // 3:] = 1.0
    h_three = h_three + (N / 2.0 - np.sum(h_three)) / N
    h_three = np.clip(h_three, 0, 1)
    inits.append(h_three)

    # Normalize sums for inits to start closer to feasible region
    processed_inits = []
    for h_init in inits:
        h_init = np.clip(h_init, 0, 1)
        diff = N/2.0 - np.sum(h_init)
        h_init += diff/N
        h_init = np.clip(h_init, 0, 1)
        processed_inits.append(h_init)
        
    # Add random restarts
    num_random_restarts = 10
    for i in range(num_random_restarts):
        if time.time() > deadline - 20: # Leave time for final checks
            break
        h_rand = np.random.uniform(0, 1, N)
        diff = N/2.0 - np.sum(h_rand)
        h_rand += diff/N
        h_rand = np.clip(h_rand, 0, 1)
        processed_inits.append(h_rand)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    for i, h_init in enumerate(processed_inits):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective, 
                h_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            if res.success:
                try_update(res.x)
            else:
                try_update(res.x)
        except Exception:
            continue

    # Final polish with higher precision if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                try_update(res.x)
            else:
                try_update(res.x)
        except Exception:
            pass

    # Final validation and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        
    return best_h

# EVOLVE-BLOCK-END
