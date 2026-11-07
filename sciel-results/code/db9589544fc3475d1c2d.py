# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100

def _eval_value(h):
    """Compute the exact objective: max correlate(h, 1-h, full) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr) / N * 2.0)

def _project(h):
    """Project h to [0,1] with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Simple iterative projection to satisfy sum and bounds
    for _ in range(20):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-12:
            break
        h = h - diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, beta):
    """
    Smooth surrogate for max(correlate(h, 1-h, full)) / N * 2.
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    # logsumexp(beta * corr) approximates beta * max(corr)
    # We want to minimize this.
    sm = logsumexp(beta * corr)
    return (sm / beta) / N * 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_h = np.full(N, 0.5)
    best_val = _eval_value(best_h)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Seed
    starts.append(np.full(N, 0.5))
    
    # 2. Random uniform
    for _ in range(10):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = _project(h_rand)
        starts.append(h_rand)
        
    # 3. Sinusoidal patterns
    for freq in [1, 2, 3, 5, 8, 12]:
        i = np.arange(N)
        h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * freq * i / N)
        h_sin = _project(h_sin)
        starts.append(h_sin)
        
    # 4. Blocky patterns
    for block_size in [2, 4, 5, 10, 20]:
        h_block = np.zeros(N)
        h_block[::2] = 1.0
        # Shift to create different phases
        shift = rng.randint(0, block_size)
        h_block = np.roll(h_block, shift)
        h_block = _project(h_block)
        starts.append(h_block)
        
    # 5. Sawtooth
    h_tri = np.linspace(0, 1, N)
    h_tri = _project(h_tri)
    starts.append(h_tri)
    
    # Optimization loop
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # We will try to optimize with a few different beta values (annealing)
    # Start with low beta for broad search, increase for precision
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for start_h in starts:
        if time.time() > deadline - 10.0:
            break
            
        current_h = start_h.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective for this beta
            def obj(h, beta=beta):
                return _soft_max_objective(h, beta)
            
            # Run SLSQP
            try:
                res = minimize(
                    obj, 
                    current_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                if res.success:
                    h_candidate = res.x
                    h_candidate = _project(h_candidate) # Ensure strict feasibility
                    v_exact = _eval_value(h_candidate)
                    if v_exact < best_val:
                        best_val = v_exact
                        best_h = h_candidate.copy()
                        current_h = h_candidate.copy()
                elif res.x is not None:
                    # Even if not successful, check the result
                    h_candidate = res.x
                    h_candidate = _project(h_candidate)
                    v_exact = _eval_value(h_candidate)
                    if v_exact < best_val:
                        best_val = v_exact
                        best_h = h_candidate.copy()
                        current_h = h_candidate.copy()
            except Exception:
                pass
                
    # Final Polish: If we have time, do a very tight local search on the best solution
    if time.time() < deadline - 2.0:
        # Use a high beta for very precise max approximation
        def obj_polish(h, beta=2000.0):
            return _soft_max_objective(h, beta)
            
        try:
            res = minimize(
                obj_polish, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                h_candidate = res.x
                h_candidate = _project(h_candidate)
                v_exact = _eval_value(h_candidate)
                if v_exact < best_val:
                    best_val = v_exact
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # Final check and return
    best_h = _project(best_h)
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
