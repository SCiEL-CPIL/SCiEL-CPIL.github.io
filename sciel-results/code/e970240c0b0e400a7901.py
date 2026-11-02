import numpy as np
import itertools
import time
from scipy.optimize import linprog

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict graph: edge if M[i,j] > 2 (incompatible)
    # We want maximum independent set: no two selected have M[i,j] > 2
    
    # Build adjacency as boolean matrix (True = conflict/incompatible)
    conflict = (M > 2).astype(np.int32)
    np.fill_diagonal(conflict, 0)
    
    # Greedy initialization with multiple orderings
    rng = np.random.default_rng(42)
    
    compat = (M <= 2).astype(np.int32)
    compat_count = compat.sum(axis=1)
    
    orderings = [np.argsort(-compat_count), np.arange(N)]
    for _ in range(50):
        orderings.append(rng.permutation(N))
    
    best_set = []
    best_size = 0
    
    for oi, order in enumerate(orderings):
        if time.time() > deadline - 10:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search with swaps to improve the greedy solution
    if time.time() < deadline - 15 and best_set:
        best_set_arr = np.array(best_set)
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set_arr] = True
        
        # Try 1-for-1 and 1-for-2 swaps
        improved = True
        swap_round = 0
        while improved and time.time() < deadline - 15:
            improved = False
            swap_round += 1
            
            unselected = np.where(~in_set)[0]
            if len(unselected) == 0:
                break
            
            # For each selected vertex, try swapping with one or two unselected
            for sel_idx in best_set_arr.copy():
                if time.time() > deadline - 15:
                    break
                
                # Try removing sel_idx and adding one unselected
                # Check if any unselected is compatible with all other selected
                other_selected = best_set_arr[best_set_arr != sel_idx]
                if len(other_selected) > 0:
                    compat_with_others = M[np.ix_(unselected, other_selected)]
                    can_replace = np.all(compat_with_others <= 2, axis=1)
                    replaceable = unselected[can_replace]
                    
                    # Prefer to add a vertex that has higher compatibility count
                    if len(replaceable) > 0:
                        # Pick the one with highest degree in the compatibility graph
                        degs = compat_count[replaceable]
                        best_repl = replaceable[np.argmax(degs)]
                        
                        # Verify: best_repl is compatible with all other selected
                        dots_check = M[best_repl, other_selected]
                        if np.all(dots_check <= 2):
                            # Perform swap
                            in_set[sel_idx] = False
                            in_set[best_repl] = True
                            best_set.remove(sel_idx)
                            best_set.append(best_repl)
                            best_set_arr = np.array(best_set)
                            improved = True
                            unselected = np.where(~in_set)[0]
                            if len(unselected) == 0:
                                break
                            continue
                
                # Try 1-for-2: remove sel_idx, add two unselected that are compatible with each other and with all other selected
                if len(other_selected) > 0:
                    compat_with_others = M[np.ix_(unselected, other_selected)]
                    can_with_others = np.all(compat_with_others <= 2, axis=1)
                    candidates_for_pair = unselected[can_with_others]
                    
                    if len(candidates_for_pair) >= 2:
                        # Find a compatible pair among candidates_for_pair
                        pair_found = False
                        for i in range(min(len(candidates_for_pair), 50)):
                            if pair_found or time.time() > deadline - 15:
                                break
                            for j in range(i + 1, min(len(candidates_for_pair), 50)):
                                if M[candidates_for_pair[i], candidates_for_pair[j]] <= 2:
                                    # Found a compatible pair
                                    v1, v2 = candidates_for_pair[i], candidates_for_pair[j]
                                    in_set[sel_idx] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    best_set.remove(sel_idx)
                                    best_set.extend([v1, v2])
                                    best_set_arr = np.array(best_set)
                                    improved = True
                                    unselected = np.where(~in_set)[0]
                                    pair_found = True
                                    break
    
    # Final local search: try to add more vectors
    if time.time() < deadline - 5 and best_set:
        best_set_arr = np.array(best_set)
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set_arr] = True
        
        unselected = np.where(~in_set)[0]
        if len(unselected) > 0:
            compat_with_selected = M[np.ix_(unselected, best_set_arr)]
            can_add = np.all(compat_with_selected <= 2, axis=1)
            addable = unselected[can_add]
            
            for idx in addable:
                if time.time() > deadline - 3:
                    break
                dots = M[idx, in_set]
                if np.all(dots <= 2):
                    best_set.append(idx)
                    in_set[idx] = True
                    best_size = len(best_set)
    
    if not best_set:
        # Fallback: use the known 220-vector D_11 kissing configuration
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[best_set].astype(np.int64)
    
    # Verify validity
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return result
