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

def greedy_sidon_search(indices, deadline):
    """Greedy heuristic with random restarts to find a large Sidon subset of indices.
    Returns the best Sidon set found."""
    best_sidon = set()
    for restart in range(50):
        if time.time() > deadline - 20.0:
            break
        # Random ordering of candidates
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        current = set()
        for idx in shuffled:
            # Check if adding idx preserves Sidon property
            test_set = list(current) + [idx]
            if is_sidon_set(test_set):
                current.add(idx)
        if len(current) > len(best_sidon):
            best_sidon = set(current)
    return list(best_sidon)

def local_search_polish(h, deadline):
    """Local search: try swapping in/out indices and adjusting weights."""
    h = h.copy()
    improved = True
    max_iter = 100
    for _ in range(max_iter):
        if time.time() > deadline - 15.0:
            break
        if not improved:
            break
        improved = False
        current_score = compute_ratio(h)
        
        # Try adding a new index with small weight
        nonzero_idx = set(np.where(h > 1e-10)[0])
        zero_idx = [i for i in range(N) if i not in nonzero_idx]
        for idx in zero_idx:
            if time.time() > deadline - 15.0:
                break
            h_test = h.copy()
            h_test[idx] = 0.01
            sc = compute_ratio(h_test)
            if sc > current_score + 1e-10:
                h = h_test
                current_score = sc
                improved = True
                break
        
        # Try removing an index
        nonzero_list = list(nonzero_idx)
        for idx in nonzero_list:
            if time.time() > deadline - 15.0:
                break
            h_test = h.copy()
            h_test[idx] = 0.0
            sc = compute_ratio(h_test)
            if sc > current_score + 1e-10:
                h = h_test
                current_score = sc
                improved = True
                break
                
        # Try adjusting weights of existing indices
        for idx in nonzero_list:
            if time.time() > deadline - 15.0:
                break
            for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                h_test = h.copy()
                h_test[idx] *= factor
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    break
            if improved:
                break
                
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin: kill at 150s, return by 140s

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

    np.random.seed(42)

    # Phase 1: Addition Graph Maximum Independent Set (Sidon Set) Search
    # Find the largest subset of [0, N) that forms a Sidon set
    all_indices = list(range(N))
    
    # Multiple random restarts for greedy Sidon search
    best_sidon = []
    for restart in range(20):
        if time.time() > deadline - 60.0:
            break
        sidon_set = greedy_sidon_search(all_indices, deadline)
        if len(sidon_set) > len(best_sidon):
            best_sidon = sidon_set
    
    # Also try some known good Sidon constructions
    known_sidon_patterns = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 11, 23],
        [0, 1, 4, 9, 16, 25, 36, 49],
    ]
    
    all_sidon_candidates = [best_sidon] + known_sidon_patterns
    
    # Evaluate all Sidon candidates
    for sidon_set in all_sidon_candidates:
        if time.time() > deadline - 50.0:
            break
        if len(sidon_set) < 2:
            continue
        h = np.zeros(N)
        for idx in sidon_set:
            if 0 <= idx < N:
                h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 2: Random sparse search for additional candidates
    num_sparse_tries = 500
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 40.0:
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
    top_k = candidates[:15]
    
    # Phase 3: Local Optimization from Top Candidates
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
        
        # Optimize weights for this fixed support
        res = minimize(objective_neg_sparse, h_sparse_start, args=(indices,), 
                       method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        h_opt_sparse = np.maximum(res.x, 0.0)
        h_opt = np.zeros(N)
        h_opt[indices] = h_opt_sparse
        
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            sc_opt = compute_ratio(h_opt)
            update_best(h_opt, sc_opt)
            
        # Apply local search polish to the best candidate
        if i == 0:
            h_polished = local_search_polish(h_opt, deadline)
            sc_polished = compute_ratio(h_polished)
            update_best(h_polished, sc_polished)

    return list(best_h)
