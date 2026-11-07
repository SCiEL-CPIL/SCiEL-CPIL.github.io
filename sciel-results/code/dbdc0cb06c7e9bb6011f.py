import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_OFFSET = 145.0  # Safety margin for 150s limit

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    
    # Simple iterative projection
    for _ in range(50):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to decrease sum
            mask = h > 1e-9
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum
            mask = h < 1 - 1e-9
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _construct_from_basis(coeffs):
    """
    Build h from a basis expansion:
    h[i] = 0.5 + sum_k coeffs[k] * basis_k(i)
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    # Up to 12 basis functions
    if n > 0:
        h += coeffs[0] * (i / N - 0.5)  # Linear ramp
    if n > 1:
        h += coeffs[1] * ((i / N - 0.5) ** 2 - 1/12)  # Quadratic
    if n > 2:
        h += coeffs[2] * np.sin(2 * np.pi * i / N)  # freq 1
    if n > 3:
        h += coeffs[3] * np.cos(2 * np.pi * i / N)
    if n > 4:
        h += coeffs[4] * np.sin(4 * np.pi * i / N)  # freq 2
    if n > 5:
        h += coeffs[5] * np.cos(4 * np.pi * i / N)
    if n > 6:
        h += coeffs[6] * np.sin(6 * np.pi * i / N)  # freq 3
    if n > 7:
        h += coeffs[7] * np.cos(6 * np.pi * i / N)
    if n > 8:
        h += coeffs[8] * np.sin(10 * np.pi * i / N)  # freq 5
    if n > 9:
        h += coeffs[9] * np.cos(10 * np.pi * i / N)
    if n > 10:
        h += coeffs[10] * np.sin(14 * np.pi * i / N)  # freq 7
    if n > 11:
        h += coeffs[11] * np.cos(14 * np.pi * i / N)
    
    return h

def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)

def _generate_starts():
    """Generate a diverse set of starting candidates."""
    starts = []
    rng = np.random.default_rng(100)
    
    # 1. Seed
    starts.append(np.full(N, 0.5))
    
    # 2. Random uniform
    for _ in range(20):
        h = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(h))
        
    # 3. Sine waves (various frequencies and phases)
    for freq in range(1, 10):
        for phase in [0, np.pi/4, np.pi/2]:
            h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            starts.append(_project_to_feasible(h))
            
    # 4. Cosine waves
    for freq in range(1, 10):
        for phase in [0, np.pi/4, np.pi/2]:
            h = 0.5 + 0.4 * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
            starts.append(_project_to_feasible(h))
            
    # 5. Square waves / Blocks
    for block_size in range(2, 11):
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h))
        
    # 6. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    return starts

def _optimize_candidate(h_start, time_budget=1.0):
    """Optimize a single candidate using SLSQP."""
    start_time = time.time()
    h_start = _project_to_feasible(h_start)
    
    def objective(h):
        return _correlation_value(h)
    
    def constraint_sum(h):
        return np.sum(h) - N / 2.0
    
    try:
        # Use SLSQP
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        # We run minimize with a wrapper that checks time if needed, 
        # but SLSQP is fast for N=100. We just limit maxiter.
        res = minimize(
            objective,
            h_start,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res.success and res.fun is not None:
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            return h_opt, val
        else:
            return h_start, _correlation_value(h_start)
            
    except Exception:
        return h_start, _correlation_value(h_start)

def _local_polish(h, max_iters=100):
    """
    Simple coordinate descent polish.
    Tries to reduce the max correlation by adjusting individual elements.
    """
    best_h = _project_to_feasible(h)
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    for it in range(max_iters):
        improved = False
        
        # Try adjusting each element with small steps
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0 or new_h[i] > 1:
                        continue
                    
                    # Adjust a compensating element j to keep sum
                    # Pick j that minimizes change in objective
                    # Simple: pick random j != i
                    j = np.random.choice([x for x in range(n) if x != i])
                    new_h[j] -= sign * d
                    
                    if new_h[j] < 0 or new_h[j] > 1:
                        continue
                    
                    # Project to ensure strict bounds and sum
                    new_h = _project_to_feasible(new_h)
                    
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-10:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        if not improved:
            break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    de_deadline = start_time + 60.0
    
    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_basis(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start SLSQP from good starting points
    starts = _generate_starts()
    starts.append(best_h.copy())
    
    start_vals = []
    for h in starts:
        v = _correlation_value(h)
        start_vals.append((v, h))
        
    start_vals.sort(key=lambda x: x[0])
    
    slsqp_deadline = start_time + 135.0
    
    for val, h in start_vals:
        if time.time() > slsqp_deadline:
            break
            
        h_opt, val_opt = _optimize_candidate(h)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
    # Phase 3: Polish the best
    if time.time() < deadline - 5.0:
        try:
            h_polished, val_polished = _local_polish(best_h, max_iters=50)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass
            
    # Final check
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return best_h
