"""Optimized sunflower-free family construction with precomputed conflicts."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 85.0  # wall-clock seconds; safety margin before 90s kill

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _all_4sets_bitmask():
    """Generate all 4-element subsets of [1,16] as bitmasks."""
    elems = list(range(16))
    result = []
    for combo in combinations(elems, 4):
        mask = 0
        for e in combo:
            mask |= (1 << e)
        result.append(mask)
    return result

def _greedy_with_conflict_check(all_sets, order, conflict_cache):
    """
    Greedy with fast conflict checking using precomputed pairwise intersections.
    conflict_cache[i][j] = intersection of all_sets[i] and all_sets[j]
    """
    chosen_indices = []
    chosen_set = set()
    
    for idx in order:
        if idx in chosen_set:
            continue
        s = all_sets[idx]
        ok = True
        n = len(chosen_indices)
        
        for i in range(n):
            ci = chosen_indices[i]
            inter_ci_s = conflict_cache[ci][idx]
            for j in range(i + 1, n):
                cj = chosen_indices[j]
                # Check if inter(ci,s) == inter(ci,cj) == inter(cj,s)
                if inter_ci_s == conflict_cache[ci][cj] and inter_ci_s == conflict_cache[cj][idx]:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            chosen_indices.append(idx)
            chosen_set.add(idx)
    
    return chosen_indices

def _fast_local_improvement(all_sets, chosen_indices, conflict_cache, max_iterations=50):
    """
    Fast 1-swap local improvement using precomputed conflicts.
    Returns improved chosen_indices or original if no improvement.
    """
    chosen_set = set(chosen_indices)
    current_size = len(chosen_indices)
    improved = False
    
    for _ in range(max_iterations):
        if _time_left() < 5.0:
            break
        
        best_swap = None
        best_new_size = current_size
        
        # Try each removal
        for rm_idx in list(chosen_set):
            trial_set = chosen_set - {rm_idx}
            trial_list = list(trial_set)
            n_trial = len(trial_list)
            
            # Try each addition
            for add_idx in range(len(all_sets)):
                if add_idx in trial_set:
                    continue
                
                # Check if adding add_idx creates sunflower with any two in trial_set
                ok = True
                for i in range(n_trial):
                    ai = trial_list[i]
                    inter_ai_add = conflict_cache[ai][add_idx]
                    for j in range(i + 1, n_trial):
                        aj = trial_list[j]
                        if inter_ai_add == conflict_cache[ai][aj] and inter_ai_add == conflict_cache[aj][add_idx]:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    # This swap is valid; since we remove 1 and add 1, size stays same
                    # But we want to find if we can get a different configuration
                    # For now, just track that a valid swap exists
                    if best_swap is None:
                        best_swap = (rm_idx, add_idx)
                    break
        
        if best_swap is not None:
            rm_idx, add_idx = best_swap
            chosen_set.remove(rm_idx)
            chosen_set.add(add_idx)
            improved = True
        else:
            break
    
    return list(chosen_set)

def _multi_start_greedy(all_sets, conflict_cache, num_starts=100, rng=None):
    """
    Run greedy multiple times with different orderings and return best result.
    """
    if rng is None:
        rng = random.Random()
    
    n_all = len(all_sets)
    best_chosen = None
    best_size = 0
    
    # First: try lexicographic order
    lex_order = list(range(n_all))
    chosen = _greedy_with_conflict_check(all_sets, lex_order, conflict_cache)
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_chosen = chosen
    
    # Try several random orderings
    for start in range(num_starts):
        if _time_left() < 10.0:
            break
        
        order = list(range(n_all))
        rng.shuffle(order)
        
        chosen = _greedy_with_conflict_check(all_sets, order, conflict_cache)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen
        
        # Try local improvement on the best so far
        if _time_left() > 15.0 and len(chosen) >= best_size:
            improved = _fast_local_improvement(all_sets, chosen, conflict_cache)
            if len(improved) > best_size:
                best_size = len(improved)
                best_chosen = improved
    
    # Final local improvement on best
    if _time_left() > 10.0 and best_chosen is not None:
        improved = _fast_local_improvement(all_sets, best_chosen, conflict_cache)
        if len(improved) > best_size:
            best_size = len(improved)
            best_chosen = improved
    
    return best_chosen

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    all_sets = _all_4sets_bitmask()
    n_all = len(all_sets)
    
    # Precompute pairwise intersections for fast conflict checking
    conflict_cache = [[0] * n_all for _ in range(n_all)]
    for i in range(n_all):
        for j in range(i + 1, n_all):
            inter = all_sets[i] & all_sets[j]
            conflict_cache[i][j] = inter
            conflict_cache[j][i] = inter
    
    rng = random.Random(42)
    best_chosen = _multi_start_greedy(all_sets, conflict_cache, num_starts=200, rng=rng)
    
    if best_chosen is None:
        best_chosen = list(range(min(10, n_all)))
    
    # Convert back to sorted lists
    result = []
    for idx in best_chosen:
        mask = all_sets[idx]
        elems = []
        for e in range(16):
            if mask & (1 << e):
                elems.append(e + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
