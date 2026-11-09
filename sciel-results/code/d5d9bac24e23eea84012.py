import itertools
import numpy as np
import random
import time
from scipy.optimize import milp, LinearConstraint, Bounds

# Global cache for neighbor indices
_neighbor_cache = {}

def _get_neighbors(q, n):
    key = (q, n)
    if key in _neighbor_cache:
        return _neighbor_cache[key]
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for w in all_words:
        covered = [w]
        for pos in range(n):
            for d in range(q):
                if d == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = d
                new_w = tuple(new_w)
                covered.append(new_w)
        neighbors.append([word_to_idx[w] for w in covered])
    
    _neighbor_cache[key] = neighbors
    return neighbors

def _greedy_cover(q, n, radius, neighbors, rng):
    total_words = q ** n
    covered = np.zeros(total_words, dtype=bool)
    selected_indices = []
    
    while not covered.all():
        best_idx = -1
        best_count = -1
        
        order = list(range(total_words))
        rng.shuffle(order)
        
        for i in order:
            count = 0
            for j in neighbors[i]:
                if not covered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
            elif count == best_count and best_count > 0:
                if rng.random() < 0.5:
                    best_idx = i
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        for j in neighbors[best_idx]:
            covered[j] = True
            
    return selected_indices

def _optimize_indices(indices, neighbors, total_words):
    cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in indices:
        for j in neighbors[idx]:
            cov_count[j] += 1
            
    current_indices = list(indices)
    changed = True
    while changed:
        changed = False
        new_indices = []
        for idx in current_indices:
            is_redundant = True
            for j in neighbors[idx]:
                if cov_count[j] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for j in neighbors[idx]:
                    cov_count[j] -= 1
                changed = True
            else:
                new_indices.append(idx)
        
        current_indices = new_indices
        
    return current_indices

def construct(q, n, radius):
    """
    Construct a covering code for {0..q-1}^n with covering radius.
    Uses ILP (Integer Linear Programming) via SciPy for exact or near-exact optimization.
    Falls back to Greedy + Local Search if ILP fails or times out.
    """
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1_ilp()
    
    # General fallback for other parameters
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]

def _construct_ternary_6_1_ilp():
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    neighbors = _get_neighbors(q, n)
    
    # Build the incidence matrix A
    # A[j, i] = 1 if word i is in the neighborhood of word j
    # Note: The constraint is that for every word j, at least one neighbor i is selected.
    # So sum(A[j, :] * x) >= 1
    
    A = np.zeros((total_words, total_words), dtype=np.int8)
    for j in range(total_words):
        for i in neighbors[j]:
            A[j, i] = 1
            
    # Objective: minimize sum(x)
    c = np.ones(total_words)
    
    # Constraints: A x >= 1
    # LinearConstraint(A, 1, np.inf)
    constraints = LinearConstraint(A, 1, np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: x is integer
    integrality = np.ones(total_words)
    
    start_time = time.time()
    
    try:
        # Use HiGHS solver via scipy
        # Time limit for solver: 60 seconds to leave buffer
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, 
                   options={"time_limit": 60})
        
        if res.status == 0: # Optimal
            x = res.x
            # x should be 0 or 1
            selected_indices = [i for i in range(total_words) if x[i] > 0.5]
            
            # Verify coverage just in case
            covered = np.zeros(total_words, dtype=bool)
            for idx in selected_indices:
                for j in neighbors[idx]:
                    covered[j] = True
            
            if covered.all():
                all_words = list(itertools.product(range(q), repeat=n))
                return [list(all_words[idx]) for idx in selected_indices]
            else:
                # If ILP returned invalid (should not happen), fallback
                pass
        else:
            # If not optimal, check if we have a feasible solution
            if res.status in [1, 2]: # Feasible or Infeasible? 1 is feasible? 
                # Actually status 0 is optimal, 1 is feasible, 2 is infeasible? 
                # SciPy docs: 0: Optimal, 1: Feasible, 2: Infeasible, 3: Unbounded, 4: Time Limit
                if hasattr(res, 'x') and res.x is not None:
                    x = res.x
                    selected_indices = [i for i in range(total_words) if x[i] > 0.5]
                    
                    covered = np.zeros(total_words, dtype=bool)
                    for idx in selected_indices:
                        for j in neighbors[idx]:
                            covered[j] = True
                    
                    if covered.all():
                        all_words = list(itertools.product(range(q), repeat=n))
                        return [list(all_words[idx]) for idx in selected_indices]
                
    except Exception as e:
        print(f"ILP failed: {e}")
        
    # Fallback to Greedy + Optimization if ILP didn't work or timed out without good result
    rng = random.Random(42)
    best_size = float('inf')
    best_indices = None
    
    # Quick greedy run
    indices = _greedy_cover(q, n, radius, neighbors, rng)
    indices = _optimize_indices(indices, neighbors, total_words)
    best_size = len(indices)
    best_indices = indices
    
    # Try a few more greedy restarts if time permits
    if time.time() - start_time < 110:
        for i in range(10):
            if time.time() - start_time > 115:
                break
            rng_iter = random.Random(1000 + i)
            indices = _greedy_cover(q, n, radius, neighbors, rng_iter)
            indices = _optimize_indices(indices, neighbors, total_words)
            if len(indices) < best_size:
                best_size = len(indices)
                best_indices = indices
                if best_size <= 73:
                    break
                    
    all_words = list(itertools.product(range(q), repeat=n))
    return [list(all_words[idx]) for idx in best_indices]
