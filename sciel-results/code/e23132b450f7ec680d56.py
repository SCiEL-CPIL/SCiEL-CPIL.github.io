import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon_search(indices, deadline, max_size=15, num_restarts=200):
    """Greedy heuristic with random restarts to find a large Sidon subset of indices.
    Returns the best Sidon set found."""
    best_sidon = set()
    best_size = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Random ordering of candidates
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        current = set()
        
        for idx in shuffled:
            # Check if adding idx preserves Sidon property
            temp = current | {idx}
            if is_sidon_set(list(temp)):
                current.add(idx)
                if len(current) > best_size:
                    best_size = len(current)
                    best_sidon = current.copy()
        
        # Early termination if we found a large enough set
        if best_size >= max_size:
            break
            
    return list(best_sidon)

def generate_sidon_sets(deadline, target_size=8, num_sets=50):
    """Generate multiple Sidon sets using greedy search."""
    all_indices = list(range(N))
    sidon_sets = []
    
    for _ in range(num_sets):
        if time.time() > deadline - 15.0:
            break
        
        sidon = greedy_sidon_search(all_indices, deadline, target_size, num_restarts=20)
        if sidon and len(sidon) >= 3:
            # Avoid duplicates
            is_dup = False
            for existing in sidon_sets:
                if set(existing) == set(sidon):
                    is_dup = True
                    break
            if not is_dup:
                sidon_sets.append(sidon)
        
        if len(sidon_sets) >= num_sets:
            break
            
    return sidon_sets

def optimize_weights_for_support(indices, deadline, max_iter=3000):
    """Optimize weights for a fixed support using Nelder-Mead."""
    def objective_neg(h_sparse):
        h = np.zeros(N)
        h[indices] = np.maximum(h_sparse, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    # Initialize with equal weights
    h_sparse_start = np.ones(len(indices)) * 0.5
    
    try:
        res = minimize(objective_neg, h_sparse_start, 
                       method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        h_opt_sparse = np.maximum(res.x, 0.0)
        h_opt = np.zeros(N)
        h_opt[indices] = h_opt_sparse
        
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            return h_opt, compute_ratio(h_opt)
    except Exception:
        pass
    
    return None, 0.0

def local_search_polish(h, deadline, max_iters=500):
    """Local search polish: try swapping or adjusting weights."""
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    
    indices = np.where(best_h > 1e-10)[0]
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        
        # Try adjusting each weight
        for idx in indices:
            old_val = best_h[idx]
            for factor in [0.5, 0.7, 0.8, 1.2, 1.5]:
                trial = best_h.copy()
                trial[idx] = old_val * factor
                trial = np.maximum(trial, 0.0)
                s = np.sum(trial)
                if s < 1e-15:
                    continue
                trial = trial / s
                sc = compute_ratio(trial)
                if sc > best_score + 1e-10:
                    best_h = trial
                    best_score = sc
                    improved = True
                    break
        
        # Try adding a new index
        if not improved and len(indices) < 20:
            for new_idx in range(N):
                if new_idx in indices:
                    continue
                trial = best_h.copy()
                trial[new_idx] = 0.1
                trial = np.maximum(trial, 0.0)
                s = np.sum(trial)
                if s < 1e-15:
                    continue
                trial = trial / s
                sc = compute_ratio(trial)
                if sc > best_score + 1e-10:
                    best_h = trial
                    best_score = sc
                    indices = np.where(best_h > 1e-10)[0]
                    improved = True
                    break
        
        # Try removing an index
        if not improved and len(indices) > 3:
            for idx in indices:
                trial = best_h.copy()
                trial[idx] = 0.0
                trial = np.maximum(trial, 0.0)
                s = np.sum(trial)
                if s < 1e-15:
                    continue
                trial = trial / s
                sc = compute_ratio(trial)
                if sc > best_score + 1e-10:
                    best_h = trial
                    best_score = sc
                    indices = np.where(best_h > 1e-10)[0]
                    improved = True
                    break
        
        if not improved:
            break
            
    return best_h, best_score

def gradient_polish(h, deadline, max_iter=2000):
    """Gradient-based polish using L-BFGS-B on the full vector."""
    def objective_neg(h_flat):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    try:
        res = minimize(objective_neg, h, 
                       method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        h_opt = np.maximum(res.x, 0.0)
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            return h_opt, compute_ratio(h_opt)
    except Exception:
        pass
    
    return h, compute_ratio(h)

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Generate Sidon sets
    sidon_sets = generate_sidon_sets(deadline, target_size=8, num_sets=30)
    
    # Phase 2: Optimize weights for each Sidon set
    candidates = []
    
    for sidon in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        h_opt, sc = optimize_weights_for_support(sidon, deadline, max_iter=2000)
        if h_opt is not None and sc > 0:
            candidates.append((sc, h_opt))
    
    # Phase 3: Also try some random sparse supports
    for _ in range(100):
        if time.time() > deadline - 25.0:
            break
        
        k = np.random.randint(4, 12)
        indices = np.random.choice(N, k, replace=False)
        weights = np.random.rand(k) * 0.5 + 0.1
        h = np.zeros(N)
        h[indices] = weights
        
        sc = compute_ratio(h)
        if sc > 0:
            candidates.append((sc, h))
    
    # Sort and keep top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:15]
    
    # Update best from candidates
    for sc, h in top_candidates:
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 4: Local search polish on top candidates
    for sc, h in top_candidates[:8]:
        if time.time() > deadline - 15.0:
            break
        
        h_polished, sc_polished = local_search_polish(h, deadline, max_iters=200)
        if sc_polished > best_score:
            best_score = sc_polished
            best_h = h_polished.copy()
    
    # Phase 5: Gradient polish on the best candidate
    if time.time() > deadline - 10.0:
        pass
    else:
        h_grad, sc_grad = gradient_polish(best_h, deadline, max_iter=1000)
        if sc_grad > best_score:
            best_score = sc_grad
            best_h = h_grad.copy()
    
    # Final local polish
    if time.time() > deadline - 3.0:
        pass
    else:
        h_final, sc_final = local_search_polish(best_h, deadline, max_iters=100)
        if sc_final > best_score:
            best_score = sc_final
            best_h = h_final.copy()
    
    return list(best_h)
