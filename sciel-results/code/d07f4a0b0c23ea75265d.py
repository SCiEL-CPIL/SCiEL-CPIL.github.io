"""Erdos minimum overlap: Low-dimensional Fourier parameterization approach."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    """Compute the objective value: max correlation / N * 2."""
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """Project h onto the affine hyperplane sum(h)=N/2 and box [0,1]."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    
    # Final check and correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _fourier_to_h(params):
    """
    Convert Fourier coefficients to a height sequence.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    We use real-valued Fourier series:
    h[i] = a0 + sum_{k=1}^K (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    
    To ensure sum(h) = N/2, we fix a0 = 0.5.
    So params are [a1, b1, ..., aK, bK].
    """
    K = len(params) // 2
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, K + 1):
        ak = params[2*(k-1)]
        bk = params[2*(k-1)+1]
        h += ak * np.cos(2 * np.pi * k * i / N) + bk * np.sin(2 * np.pi * k * i / N)
    return h

def _objective_fourier(params):
    """Objective function for Fourier parameter optimization."""
    h = _fourier_to_h(params)
    # Enforce box constraints [0,1] by clipping, then fix sum
    # Note: Clipping distorts the Fourier structure, but we need valid h for scoring.
    # A better approach for optimization would be soft constraints, but clipping 
    # followed by sum-fixing is robust for evaluation.
    h_clipped = np.clip(h, 0, 1)
    h_fixed = _fix_sum(h_clipped)
    
    if not _is_valid(h_fixed):
        return 10.0 + np.sum(np.abs(h_fixed - np.clip(h_fixed, 0, 1)))
    
    val = _score(h_fixed)
    
    # Penalty if the clipping/fixing changed the structure significantly?
    # For now, just return the score. The optimizer will learn to stay within bounds.
    return val

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Strategy: Optimize over Fourier coefficients
    # We will try different numbers of harmonics K
    # K=1 is a sine wave, K=2 adds second harmonic, etc.
    
    # Start with a few random initializations for the Fourier params
    rng = np.random.RandomState(42)
    
    max_K = 10 # Use first 10 harmonics (20 parameters)
    
    # Initial guess: all zeros (which is h=0.5)
    # But we want to explore non-trivial structures.
    
    # We will use differential_evolution for global search over Fourier params.
    # Bounds for Fourier coefficients: [-0.5, 0.5] is a reasonable start, 
    # maybe slightly wider to allow peaks, but sum constraint fixes mean.
    # If h[i] = 0.5 + A*sin(...), A can be up to 0.5 to stay in [0,1] ideally.
    # Let's use [-0.6, 0.6] to allow some flexibility.
    
    bounds = [(-0.6, 0.6)] * (2 * max_K)
    
    # Differential evolution settings
    # popsize=15, maxiter=100, tol=1e-6
    # This should be fast enough for 20 dimensions.
    
    try:
        # Run DE
        de_result = differential_evolution(
            _objective_fourier,
            bounds=bounds,
            seed=42,
            maxiter=150,
            tol=1e-8,
            popsize=20,
            polish=True,
            init='sobol'
        )
        
        if de_result.success:
            h_opt = _fourier_to_h(de_result.x)
            h_opt = np.clip(h_opt, 0, 1)
            h_opt = _fix_sum(h_opt)
            if _is_valid(h_opt):
                s_opt = _score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
    except Exception:
        pass
    
    # Refine the best solution found by DE using local optimization (L-BFGS-B)
    # This helps polish the Fourier coefficients for better precision.
    if not deadline_check():
        # Get the best params from DE (if it ran) or start from zero
        # We need to recover params. DE returns x.
        # If DE failed, we try a few random starts for local opt.
        
        # Let's try local optimization from the DE result if available, 
        # or from a few random Fourier configs.
        
        candidates = []
        if 'de_result' in locals() and de_result.success:
            candidates.append(de_result.x)
        
        # Add some random candidates
        for _ in range(5):
            candidates.append(rng.uniform(-0.5, 0.5, size=2*max_K))
        
        for params in candidates:
            if deadline_check():
                break
            
            try:
                # Local optimization
                local_res = minimize(
                    _objective_fourier,
                    params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                
                if local_res.success:
                    h_opt = _fourier_to_h(local_res.x)
                    h_opt = np.clip(h_opt, 0, 1)
                    h_opt = _fix_sum(h_opt)
                    if _is_valid(h_opt):
                        s_opt = _score(h_opt)
                        if s_opt < best_score:
                            best_score = s_opt
                            best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final refinement: Coordinate descent on the full h vector
    # This allows escaping the Fourier subspace if beneficial.
    if not deadline_check():
        h = best_h.copy()
        current_score = _score(h)
        
        # Simple random perturbation + projection
        for _ in range(500):
            if deadline_check():
                break
            
            idx = rng.randint(N)
            delta = rng.uniform(-0.05, 0.05)
            h_new = h.copy()
            h_new[idx] += delta
            h_new = np.clip(h_new, 0, 1)
            h_new = _fix_sum(h_new)
            
            if _is_valid(h_new):
                s_new = _score(h_new)
                if s_new < current_score:
                    current_score = s_new
                    h = h_new
        
        if current_score < best_score:
            best_score = current_score
            best_h = h
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
