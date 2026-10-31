import numpy as np
import itertools
import time
import sys

D = 11

def generate_weight_4_vectors():
    """Generate all weight-4 vectors in R^11 (exactly 4 non-zero entries, each ±1)."""
    vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def greedy_mis(candidates, deadline, seed=0):
    """
    Greedy Maximum Independent Set on conflict graph.
    Two vectors conflict if dot product > 2 (for weight-4 vectors with norm²=4).
    Returns the selected set.
    """
    n = len(candidates)
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Precompute dot product matrix (conflict if dot > 2)
    dots = candidates @ candidates.T
    # Conflict: dot > 2, i.e., dot >= 3
    conflict = dots > 2
    
    # Remove self-loops
    np.fill_diagonal(conflict, False)
    
    # Degree of each node
    degrees = np.sum(conflict, axis=1)
    
    # Order by degree (ascending) for greedy
    order = np.argsort(degrees, kind='stable')
    
    # Apply random perturbation based on seed
    if seed > 0:
        rng = np.random.default_rng(seed)
        # Add small random noise to break ties
        perturbed_degrees = degrees + rng.random(n) * 0.1
        order = np.argsort(perturbed_degrees, kind='stable')
    
    selected = []
    selected_set = set()
    available = set(range(n))
    
    for idx in order:
        if time.time() > deadline:
            break
        if idx not in available:
            continue
        
        # Check if idx conflicts with any already selected
        can_add = True
        for s in selected:
            if conflict[idx, s]:
                can_add = False
                break
        
        if can_add:
            selected.append(idx)
            selected_set.add(idx)
            # Remove all neighbors of idx from available
            neighbors = np.where(conflict[idx])[0]
            for nb in neighbors:
                available.discard(int(nb))
            available.discard(int(idx))
    
    return candidates[np.array(selected, dtype=np.int64)]

def local_improve(selected, candidates, deadline):
    """
    Try to improve the selected set by swapping or adding vectors.
    """
    if len(selected) == 0:
        return selected
    
    # Build a lookup: which candidate indices are in selected
    selected_set = set(map(tuple, selected.tolist()))
    
    # For each non-selected candidate, check if it can replace a selected one
    # or be added
    n_candidates = len(candidates)
    
    # Compute dot products of selected with all candidates
    dots_sel = selected @ candidates.T  # shape (len(selected), n_candidates)
    
    # A candidate j can be added if dot with all selected <= 2
    # dots_sel[:, j] gives dots of all selected with candidate j
    # We need all dots_sel[:, j] <= 2
    max_dots = np.max(dots_sel, axis=0)
    addable = max_dots <= 2
    
    # Try to add any addable candidates that aren't already selected
    added = 0
    for j in range(n_candidates):
        if time.time() > deadline:
            break
        if addable[j] and tuple(candidates[j].tolist()) not in selected_set:
            selected = np.vstack([selected, candidates[j]])
            selected_set.add(tuple(candidates[j].tolist()))
            added += 1
            # Recompute dots for new selection
            dots_sel = selected @ candidates.T
            max_dots = np.max(dots_sel, axis=0)
            addable = max_dots <= 2
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin before 150s kill
    
    # Generate all weight-4 vectors
    candidates = generate_weight_4_vectors()
    n = len(candidates)
    
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    best_result = np.empty((0, D), dtype=np.int64)
    best_size = 0
    
    # Multiple random restarts with different orderings
    for seed in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_mis(candidates, deadline, seed=seed)
        
        # Local improvement
        result = local_improve(result, candidates, deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    return best_result

# EVOLVE-BLOCK-END
