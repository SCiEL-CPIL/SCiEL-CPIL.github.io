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
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_size = N
    best_codewords = None
    rng = np.random.RandomState(42)
    
    # Simulated annealing with binary codeword inclusion
    # State: binary vector x of length N, x[c] = 1 if codeword c is included
    # Objective: minimize (sum(x) + lambda * uncovered_count)
    # where uncovered_count = number of points not covered by any selected codeword
    
    # First, get a good initial solution using greedy
    initial_solution = greedy_covering(cov, N, rng)
    best_size = len(initial_solution)
    best_codewords = [all_words[c].tolist() for c in initial_solution]
    
    # Simulated annealing parameters
    T0 = 5.0
    T_end = 0.01
    cooling_rate = 0.9995
    lambda_weight = 10.0  # Weight for uncovered points penalty
    
    x = np.zeros(N, dtype=np.int8)
    x[list(initial_solution)] = 1
    
    # Compute initial coverage and uncovered count
    coverage_count = np.zeros(N, dtype=np.int32)
    for c in initial_solution:
        coverage_count += cov[c]
    uncovered = (coverage_count == 0).sum()
    current_obj = len(initial_solution) + lambda_weight * uncovered
    
    T = T0
    steps = 0
    max_steps = 500000
    
    while T > T_end and steps < max_steps:
        if time.time() > deadline:
            break
        
        # Choose a random codeword to toggle
        c = rng.randint(N)
        
        if x[c] == 1:
            # Try removing codeword c
            # Check if c is the only cover for any point
            newly_uncovered = (cov[c] & (coverage_count == 1)).sum()
            delta_obj = -1 + lambda_weight * newly_uncovered
        else:
            # Try adding codeword c
            # Count how many new points it covers
            newly_covered = (cov[c] & (coverage_count == 0)).sum()
            delta_obj = 1 - lambda_weight * newly_covered
        
        # Acceptance criterion
        if delta_obj < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_obj / T)
        
        if accept:
            # Update state
            x[c] = 1 - x[c]
            if x[c] == 1:
                # Added codeword
                coverage_count += cov[c]
                uncovered -= newly_covered
            else:
                # Removed codeword
                coverage_count -= cov[c]
                uncovered += newly_uncovered
            
            current_obj += delta_obj
            
            # Update best
            if uncovered == 0 and current_obj < best_size:
                best_size = int(current_obj)
                best_codewords = [all_words[c].tolist() for c in np.where(x == 1)[0]]
        
        steps += 1
        T *= cooling_rate
    
    # Final local search: try removing redundant codewords from best solution
    if best_codewords is not None:
        cw_set = list(set([c for c in np.where(x == 1)[0]]))
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(cw_set)):
                temp_cw = [c for j, c in enumerate(cw_set) if j != i]
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

def greedy_covering(cov, N, rng):
    """Greedy set cover with NumPy vectorization."""
    is_covered = np.zeros(N, dtype=np.int8)
    uncovered_count = N
    selected = []
    
    while uncovered_count > 0:
        # Compute gain for each candidate: number of currently uncovered points it covers
        gain = cov @ (1 - is_covered)
        
        # Pick the center with max gain (randomize ties)
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
    
    return selected

def is_valid_covering(codeword_indices, cov, N):
    """Check if the given codeword indices cover all N points."""
    if len(codeword_indices) == 0:
        return False
    covered = np.zeros(N, dtype=np.int8)
    for c in codeword_indices:
        covered |= cov[c]
    return np.all(covered)
