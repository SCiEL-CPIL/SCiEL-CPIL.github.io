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

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def local_search_swap(h, indices, max_swaps=100):
    """Try swapping elements in the support to reduce max convolution."""
    best_h = h.copy()
    best_score = compute_ratio(h)
    current_indices = list(indices)
    
    for swap_iter in range(max_swaps):
        improved = False
        for i in range(len(current_indices)):
            for j in range(N):
                if j in current_indices:
                    continue
                new_indices = current_indices.copy()
                new_indices[i] = j
                new_indices = sorted(new_indices)
                
                if not is_sidon_set(new_indices):
                    continue
                
                new_h = np.zeros(N)
                for idx in new_indices:
                    new_h[idx] = 1.0
                sc = compute_ratio(new_h)
                if sc > best_score + 1e-12:
                    best_h = new_h
                    best_score = sc
                    current_indices = new_indices
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    # Phase 1: Generate and evaluate Sidon sets
    sidon_sets = generate_sidon_sets()
    
    for s in sidon_sets:
        if time.time() > deadline - 40.0:
            break
        h = np.zeros(N)
        for idx in s:
            h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 1b: Random sparse search
    np.random.seed(42)
    num_sparse_tries = 1000
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 30.0:
            break
        k = np.random.randint(3, 14)
        indices = np.random.choice(N, k, replace=False)
        weights = np.random.rand(k)
        h = np.zeros(N)
        h[indices] = weights
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Sort candidates by score and keep top K for local refinement
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:30]
    
    # Phase 2: Local optimization from Top Candidates
    def objective_neg_sparse(h_sparse, indices):
        h = np.zeros(N)
        h[indices] = h_sparse
        h = np.maximum(h, 0.0)
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

    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 15.0:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
        
        h_sparse_start = h_start[indices]
        
        res = minimize(objective_neg_sparse, h_sparse_start, args=(indices,), 
                       method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        h_opt_sparse = np.maximum(res.x, 0.0)
        h_opt = np.zeros(N)
        h_opt[indices] = h_opt_sparse
        
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            sc_opt = compute_ratio(h_opt)
            update_best(h_opt, sc_opt)
            
        # Try local search with swaps on the best few
        if i < 5:
            if time.time() > deadline - 10.0:
                break
            h_swapped, sc_swapped = local_search_swap(h_opt, indices, max_swaps=100)
            if sc_swapped > best_score:
                update_best(h_swapped, sc_swapped)
                new_indices = np.where(h_swapped > 1e-10)[0]
                if len(new_indices) > 0:
                    h_sparse_swapped = h_swapped[new_indices]
                    res2 = minimize(objective_neg_sparse, h_sparse_swapped, args=(new_indices,), 
                                   method='Nelder-Mead', 
                                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    h_opt2_sparse = np.maximum(res2.x, 0.0)
                    h_opt2 = np.zeros(N)
                    h_opt2[new_indices] = h_opt2_sparse
                    s2 = np.sum(h_opt2)
                    if s2 > 1e-15:
                        h_opt2 = h_opt2 / s2
                        sc_opt2 = compute_ratio(h_opt2)
                        update_best(h_opt2, sc_opt2)

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        best_indices = np.where(best_h > 1e-10)[0]
        if len(best_indices) > 0:
            best_sparse = best_h[best_indices]
            res_final = minimize(objective_neg_sparse, best_sparse, args=(best_indices,), 
                                method='Nelder-Mead', 
                                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_final_sparse = np.maximum(res_final.x, 0.0)
            h_final = np.zeros(N)
            h_final[best_indices] = h_final_sparse
            s_final = np.sum(h_final)
            if s_final > 1e-15:
                h_final = h_final / s_final
                sc_final = compute_ratio(h_final)
                update_best(h_final, sc_final)

    return list(best_h)
