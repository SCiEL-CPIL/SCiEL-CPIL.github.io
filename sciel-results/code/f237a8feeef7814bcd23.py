"""Improved program for kissing number in dimension 11."""
import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Generate all weight-3 vectors with entries in {-1, 0, 1}
    vectors = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    n = len(vectors)
    vecs = np.array(vectors, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(vecs ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # For weight-3 vectors, all have norm^2 = 3
    # Distance^2 between v_i and v_j = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # = 6 - 2*v_i.v_j
    # Need distance >= max_norm = sqrt(3)
    # So distance^2 >= 3
    # 6 - 2*dot >= 3 => dot <= 1.5 => dot <= 1 (integer)
    # Also need distance >= min possible, but max_norm is the constraint
    # Actually: min pairwise distance >= max vector norm
    # max vector norm = sqrt(3) for all weight-3 vectors
    # So we need ||v_i - v_j|| >= sqrt(3) for all pairs
    # ||v_i - v_j||^2 >= 3
    # 6 - 2*v_i.v_j >= 3
    # v_i.v_j <= 1.5, so v_i.v_j <= 1
    
    # Build conflict: vectors conflict if |v_i.v_j| >= 2 (since dot must be <= 1)
    # Wait, dot can be negative too. Let's check:
    # v_i.v_j <= 1 is the constraint (no lower bound issue since distance grows as dot decreases)
    # Actually if dot is very negative, distance is large, which is fine.
    # So conflict only when v_i.v_j > 1, i.e., v_i.v_j >= 2
    
    # Compute dot product matrix
    dots = vecs @ vecs.T
    
    # Conflict: dots[i,j] >= 2 for i != j
    conflict = (dots >= 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy independent set with strategic ordering
    # Try multiple orderings to find good independent sets
    
    def greedy_independent_set(order):
        selected = set()
        for idx in order:
            # Check if idx conflicts with any selected
            if idx not in selected:
                # Check conflicts with already selected
                if all(conflict[idx, s] == 0 for s in selected):
                    selected.add(idx)
        return selected
    
    # Strategy 1: Order by degree (fewest conflicts first)
    degrees = np.sum(conflict, axis=1)
    order1 = np.argsort(degrees)
    result1 = greedy_independent_set(order1)
    
    if time.time() > deadline - 30:
        return vecs[list(result1)]
    
    # Strategy 2: Randomized greedy with multiple restarts
    best_result = result1
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        # Random order with some bias toward low-degree nodes
        weights = 1.0 / (degrees + 1.0)
        weights /= np.sum(weights)
        order = rng.choice(n, size=n, replace=False, p=weights)
        result = greedy_independent_set(order)
        
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > deadline - 10:
        return vecs[list(best_result)]
    
    # Strategy 3: Local search - try to add more vectors or swap
    # Convert to list for easier manipulation
    best_list = list(best_result)
    best_set = set(best_list)
    
    # Try adding vectors that conflict with fewest selected
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Find vectors not in best_set with minimum conflicts
        candidates = [i for i in range(n) if i not in best_set]
        if not candidates:
            break
        
        min_conflicts = float('inf')
        best_candidate = None
        for c in candidates:
            conflicts = sum(1 for s in best_set if conflict[c, s] == 1)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_candidate = c
        
        if min_conflicts == 0:
            best_set.add(best_candidate)
            best_list.append(best_candidate)
            improved = True
        
        if not improved:
            # Try swap: remove one vector, add two
            break
    
    return vecs[list(best_set)]
# EVOLVE-BLOCK-END
