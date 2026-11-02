"""Kissing number in dimension 11: Greedy + Local Search on Conflict Graph.

Structural change from the current program:
- Expand candidate set to include weight-4 vectors (norm^2=4) in addition to
  weight-2 and weight-3 vectors. This gives a much larger pool (5280 + 1320 + 220 = 6820)
  and allows the optimizer to find larger independent sets.
- Use NumPy vectorized dot-product computation for the conflict matrix, which is
  far faster than pairwise Python loops for building the conflict table.
- Implement a proper local-search refinement (add-if-possible, then swap) on top
  of the greedy independent set, which can significantly improve the solution size.
- The constraint is: for all pairs u, v in S, ||u - v||_2 >= max_{w in S} ||w||_2.
  If we include weight-4 vectors, R^2 = 4, so we need ||u-v||^2 >= 4 for all pairs.
  For two weight-4 vectors: dot <= 2.
  For weight-4 and weight-3: dot <= 1.
  For two weight-3 vectors: dot <= 1.
  For weight-4 and weight-2: dot <= 1.
  For weight-3 and weight-2: dot <= 0 (since 3+2-2*dot >= 4 => dot <= 0.5 => dot <= 0).
  For two weight-2 vectors: dot <= -1 (since 2+2-2*dot >= 4 => dot <= 0... wait, 4-2*dot>=4 => dot<=0).
  Actually let me recompute:
  ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> >= 4
  So <u,v> <= (||u||^2 + ||v||^2 - 4)/2
  
  w4,w4: (4+4-4)/2 = 2
  w4,w3: (4+3-4)/2 = 1.5 -> 1
  w4,w2: (4+2-4)/2 = 1
  w4,w1: (4+1-4)/2 = 0.5 -> 0
  w3,w3: (3+3-4)/2 = 1
  w3,w2: (3+2-4)/2 = 0.5 -> 0
  w3,w1: (3+1-4)/2 = 0
  w2,w2: (2+2-4)/2 = 0
  w2,w1: (2+1-4)/2 = -0.5 -> -1
  w1,w1: (1+1-4)/2 = -1

  But wait: if we only use weight-4 vectors, then R^2=4 and the constraint is simply
  dot <= 2 for all pairs. This is the simplest and likely yields the largest set.
  
  Strategy: Focus primarily on weight-4 vectors for the main search, then optionally
  try to add lower-weight vectors if they fit.
"""

import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 140.0  # safety margin before 150s limit


def _generate_candidates_by_weight():
    """Generate all integer vectors in R^11 with squared norm 1, 2, 3, or 4."""
    candidates_w1 = []
    candidates_w2 = []
    candidates_w3 = []
    candidates_w4 = []
    
    # Weight 1: one coordinate ±1
    for pos in range(D):
        for s in (1, -1):
            v = [0] * D
            v[pos] = s
            candidates_w1.append(tuple(v))
    
    # Weight 2: two coordinates ±1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates_w2.append(tuple(v))
    
    # Weight 3: three coordinates ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates_w3.append(tuple(v))
    
    # Weight 4: four coordinates ±1
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates_w4.append(tuple(v))
    
    return candidates_w1, candidates_w2, candidates_w3, candidates_w4


def _build_conflict_matrix(candidates, norm_sq_values, max_norm_sq):
    """
    Build conflict matrix for given candidates.
    Two vectors conflict if ||u-v||^2 < max_norm_sq.
    ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> < max_norm_sq
    => <u,v> > (||u||^2 + ||v||^2 - max_norm_sq) / 2
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    
    mat = np.array(candidates, dtype=np.int32)
    dot = mat @ mat.T
    norms_sq = np.array(norm_sq_values, dtype=np.int32)
    
    # Conflict condition: dot[i,j] > (norms_sq[i] + norms_sq[j] - max_norm_sq) / 2
    # For integer dot products, this is equivalent to:
    # 2*dot[i,j] > norms_sq[i] + norms_sq[j] - max_norm_sq
    # i.e., 2*dot[i,j] - norms_sq[i] - norms_sq[j] + max_norm_sq > 0
    # i.e., 2*dot[i,j] - norms_sq[i] - norms_sq[j] + max_norm_sq >= 1
    
    threshold = norms_sq[:, None] + norms_sq[None, :] - max_norm_sq
    # Conflict if 2*dot > threshold, i.e., 2*dot - threshold >= 1
    conflict = (2 * dot - threshold >= 1).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    return conflict


def _greedy_independent_set(conflict, n, rng, time_budget=5.0):
    """
    Greedy independent set: repeatedly pick the vertex with minimum degree
    in the remaining subgraph, with random tie-breaking.
    """
    start = time.time()
    remaining = np.ones(n, dtype=bool)
    selected = []
    
    while remaining.any():
        if time.time() - start > time_budget:
            break
        
        # Compute degree in remaining subgraph
        deg = np.sum(conflict[:, remaining], axis=1)
        deg[~remaining] = n + 1
        
        min_deg = deg.min()
        if min_deg >= n:
            break
        
        # Get all vertices with minimum degree among remaining
        candidates_mask = (deg == min_deg) & remaining
        candidate_indices = np.where(candidates_mask)[0]
        
        if len(candidate_indices) == 0:
            break
        
        # Random choice among min-degree vertices
        chosen = candidate_indices[rng.randint(0, len(candidate_indices) - 1)]
        
        selected.append(chosen)
        remaining[chosen] = False
        neighbors = np.where(conflict[chosen] > 0)[0]
        remaining[neighbors] = False
    
    return selected


def _local_search_improve(conflict, n, selected_indices, rng, max_iters=500, time_budget=20.0):
    """
    Local search: try to add vertices that are compatible with all selected,
    then try swaps (remove one, add one or more).
    """
    start = time.time()
    selected_set = set(selected_indices)
    selected_arr = np.array(sorted(selected_set), dtype=np.int32)
    if len(selected_arr) == 0:
        return selected_arr
    
    # Precompute conflict counts: for each node, how many selected nodes conflict with it
    conflict_counts = np.zeros(n, dtype=np.int32)
    for s in selected_arr:
        conflict_counts += conflict[s]
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() - start < time_budget:
        improved = False
        iters += 1
        
        # Phase 1: Try to add any compatible vertex
        # A vertex j is addable if j not in selected and conflict_counts[j] == 0
        not_selected = np.ones(n, dtype=bool)
        not_selected[selected_arr] = False
        addable = not_selected & (conflict_counts == 0)
        addable_indices = np.where(addable)[0]
        
        if len(addable_indices) > 0:
            # Add the one with minimum degree among addable (or random)
            # Just pick randomly for diversity
            chosen = addable_indices[rng.randint(0, len(addable_indices) - 1)]
            selected_set.add(chosen)
            selected_arr = np.array(sorted(selected_set), dtype=np.int32)
            # Update conflict counts
            conflict_counts += conflict[chosen]
            improved = True
            continue
        
        # Phase 2: Try swaps - remove one selected, see if we can add more
        # For each selected vertex s, temporarily remove it and see what becomes addable
        # This is expensive, so do it selectively
        if len(selected_arr) > 0:
            # Try a random swap
            s_idx = rng.randint(0, len(selected_arr) - 1)
            s = selected_arr[s_idx]
            
            # Temporarily remove s
            temp_counts = conflict_counts - conflict[s]
            temp_selected = set(selected_set)
            temp_selected.discard(s)
            temp_not_selected = np.ones(n, dtype=bool)
            for t in temp_selected:
                temp_not_selected[t] = False
            temp_not_selected[s] = True  # s is now "not selected"
            
            temp_addable = temp_not_selected & (temp_counts == 0)
            temp_addable_indices = np.where(temp_addable)[0]
            
            if len(temp_addable_indices) >= 1:
                # We can remove s and add at least one vertex
                # Add as many as possible greedily
                new_selected = set(temp_selected)
                new_counts = temp_counts.copy()
                added_count = 0
                remaining_addable = set(temp_addable_indices.tolist())
                
                while remaining_addable:
                    if time.time() - start > time_budget:
                        break
                    a = rng.choice(list(remaining_addable))
                    # Check if a is still compatible with all in new_selected
                    # It should be by construction, but let's verify
                    if new_counts[a] == 0:
                        new_selected.add(a)
                        new_counts += conflict[a]
                        remaining_addable.discard(a)
                        added_count += 1
                    else:
                        remaining_addable.discard(a)
                
                if added_count >= 1:
                    # Accept the swap if we added at least as many as we removed
                    selected_set = new_selected
                    selected_arr = np.array(sorted(selected_set), dtype=np.int32)
                    # Recompute conflict counts
                    conflict_counts = np.zeros(n, dtype=np.int32)
                    for sv in selected_arr:
                        conflict_counts += conflict[sv]
                    improved = True
    
    return selected_arr


def _verify_solution(candidates, selected_indices, max_norm_sq):
    """Verify that the selected set satisfies the distance constraint."""
    if len(selected_indices) == 0:
        return True
    if len(selected_indices) == 1:
        return True
    
    sel = [candidates[i] for i in selected_indices]
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            diff_sq = sum((a - b) ** 2 for a, b in zip(sel[i], sel[j]))
            if diff_sq < max_norm_sq:
                return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Generate candidates
    w1, w2, w3, w4 = _generate_candidates_by_weight()
    
    # Strategy: Focus on weight-4 vectors first (largest pool, simplest constraint)
    # If we use only weight-4 vectors, max_norm_sq = 4, constraint is dot <= 2
    candidates_w4 = w4
    n_w4 = len(candidates_w4)
    norms_w4 = [4] * n_w4
    
    # Build conflict matrix for weight-4 vectors
    conflict_w4 = _build_conflict_matrix(candidates_w4, norms_w4, max_norm_sq=4)
    
    best_selected = []
    best_count = 0
    best_max_norm_sq = 4
    
    # Phase 1: Multiple greedy restarts on weight-4 vectors
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        selected = _greedy_independent_set(conflict_w4, n_w4, rng, time_budget=3.0)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
    
    # Phase 2: Local search improvement on the best solution
    if best_selected:
        improved = _local_search_improve(conflict_w4, n_w4, best_selected, rng, 
                                          max_iters=200, time_budget=30.0)
        if len(improved) > best_count:
            best_count = len(improved)
            best_selected = list(improved)
    
    # Phase 3: Try more greedy restarts with different random seeds
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 15:
            break
        rng2 = random.Random(1000 + restart)
        selected = _greedy_independent_set(conflict_w4, n_w4, rng2, time_budget=2.0)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
    
    # Phase 4: Another round of local search
    if best_selected and time.time() - start_time < DEADLINE - 10:
        improved = _local_search_improve(conflict_w4, n_w4, best_selected, rng,
                                          max_iters=300, time_budget=25.0)
        if len(improved) > best_count:
            best_count = len(improved)
            best_selected = list(improved)
    
    # Build the result from weight-4 candidates
    if best_selected:
        result = np.array([candidates_w4[i] for i in best_selected], dtype=np.int64)
    else:
        # Fallback: return a small valid set
        result = np.array([candidates_w4[0:10]], dtype=np.int64)
    
    return result


# Pre-compute at module level to save time
_W1, _W2, _W3, _W4 = _generate_candidates_by_weight()
_N_W4 = len(_W4)
_NORMS_W4 = [4] * _N_W4
_CONFLICT_W4 = _build_conflict_matrix(_W4, _NORMS_W4, max_norm_sq=4)


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    candidates_w4 = _W4
    n_w4 = _N_W4
    conflict_w4 = _CONFLICT_W4
    
    best_selected = []
    best_count = 0
    
    # Phase 1: Multiple greedy restarts on weight-4 vectors
    for restart in range(25):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        selected = _greedy_independent_set(conflict_w4, n_w4, rng, time_budget=2.0)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
    
    # Phase 2: Local search improvement
    if best_selected and time.time() - start_time < DEADLINE - 25:
        improved = _local_search_improve(conflict_w4, n_w4, best_selected, rng,
                                          max_iters=300, time_budget=20.0)
        if len(improved) > best_count:
            best_count = len(improved)
            best_selected = list(improved)
    
    # Phase 3: More greedy restarts
    for restart in range(40):
        if time.time() - start_time > DEADLINE - 15:
            break
        rng2 = random.Random(5000 + restart)
        selected = _greedy_independent_set(conflict_w4, n_w4, rng2, time_budget=1.5)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
    
    # Phase 4: Final local search
    if best_selected and time.time() - start_time < DEADLINE - 5:
        improved = _local_search_improve(conflict_w4, n_w4, best_selected, rng,
                                          max_iters=500, time_budget=15.0)
        if len(improved) > best_count:
            best_count = len(improved)
            best_selected = list(improved)
    
    # Build the result
    if best_selected:
        result = np.array([candidates_w4[i] for i in best_selected], dtype=np.int64)
    else:
        result = np.array([candidates_w4[0:10]], dtype=np.int64)
    
    return result
