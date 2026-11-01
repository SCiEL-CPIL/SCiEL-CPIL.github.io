import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return solve_3_6_1()
    # Fallback: return all words (valid but large)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def solve_3_6_1():
    q = 3
    n = 6
    N = q ** n  # 729
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute spheres: for each center, list of indices within Hamming distance 1
    # Sphere size = 1 + n*(q-1) = 1 + 6*2 = 13
    sphere_indices = np.zeros((N, 13), dtype=np.int32)
    for i in range(N):
        w = all_words[i]
        sphere_indices[i, 0] = i
        idx = 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Convert to base-3 index
                idx_val = 0
                for j in range(n):
                    idx_val = idx_val * 3 + new_w[j]
                sphere_indices[i, idx] = idx_val
                idx += 1
    
    # Coverage matrix: cov[c, p] = 1 if center c covers point p
    # Shape: (N, N) = (729, 729)
    cov = np.zeros((N, N), dtype=np.int8)
    for c in range(N):
        for p in sphere_indices[c]:
            cov[c, p] = 1
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    best_size = N
    best_codewords = None
    rng = np.random.RandomState(42)
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Greedy set cover with NumPy vectorization
        is_covered = np.zeros(N, dtype=np.int8)
        uncovered_count = N
        selected = []
        
        while uncovered_count > 0:
            # Compute gain for each candidate: number of currently uncovered points it covers
            # gain[c] = sum over p where not covered of cov[c, p]
            # Vectorized: gain = cov @ (1 - is_covered)
            gain = cov @ (1 - is_covered)
            
            # Pick the center with max gain (tie-break by index for determinism, or randomize)
            max_gain = np.max(gain)
            candidates = np.where(gain == max_gain)[0]
            
            if len(candidates) == 0 or max_gain == 0:
                break
            
            # Random tie-breaking to explore different solutions
            c = rng.choice(candidates)
            
            selected.append(c)
            
            # Update coverage
            newly_covered = cov[c] & (1 - is_covered)
            new_count = np.sum(newly_covered)
            is_covered |= cov[c]
            uncovered_count -= new_count
        
        if len(selected) < best_size:
            # Local search: try removing redundant codewords
            cw_set = list(set(selected))
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(cw_set)):
                    temp_cw = [c for j, c in enumerate(cw_set) if j != i]
                    # Check if still valid
                    if is_valid_covering(np.array(temp_cw), cov, N):
                        cw_set = temp_cw
                        improved = True
                        break
            
            if len(cw_set) < best_size:
                best_size = len(cw_set)
                best_codewords = [all_words[c].tolist() for c in cw_set]
    
    # Fallback if no improvement found
    if best_codewords is None:
        # Use a known construction: tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    return best_codewords

def is_valid_covering(codeword_indices, cov, N):
    """Check if the given codeword indices cover all N points."""
    if len(codeword_indices) == 0:
        return False
    covered = np.zeros(N, dtype=np.int8)
    for c in codeword_indices:
        covered |= cov[c]
    return np.all(covered)
