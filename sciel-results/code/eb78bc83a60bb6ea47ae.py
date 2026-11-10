import time
import random
import numpy as np

N = 256

def _has_4ap_fast(selected_bool, x):
    """Check if adding x to selected_bool creates a 4-AP."""
    # x is 1st term: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if selected_bool[x+d] and selected_bool[x+2*d] and selected_bool[x+3*d]:
            return True
    # x is 2nd term: x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if selected_bool[x-d] and selected_bool[x+d] and selected_bool[x+2*d]:
            return True
    # x is 3rd term: x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if selected_bool[x-2*d] and selected_bool[x-d] and selected_bool[x+d]:
            return True
    # x is 4th term: x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if selected_bool[x-3*d] and selected_bool[x-2*d] and selected_bool[x-d]:
            return True
    return False

def _count_3aps_fast(selected_bool):
    """Count 3-APs using boolean array."""
    count = 0
    for a in range(1, N+1):
        if not selected_bool[a]:
            continue
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            if selected_bool[a+d] and selected_bool[a+2*d]:
                count += 1
    return count

def _delta_3aps(selected_bool, x):
    """Count 3-APs that involve x and would be created by adding x."""
    delta = 0
    # x is 1st term: x, x+d, x+2d
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        if selected_bool[x+d] and selected_bool[x+2*d]:
            delta += 1
    # x is 2nd term: x-d, x, x+d
    max_d = min(x - 1, N - x)
    for d in range(1, max_d + 1):
        if selected_bool[x-d] and selected_bool[x+d]:
            delta += 1
    # x is 3rd term: x-2d, x-d, x
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        if selected_bool[x-2*d] and selected_bool[x-d]:
            delta += 1
    return delta

def _greedy_construct(rng, alpha=0.5):
    """GRASP construction with randomness controlled by alpha."""
    selected = np.zeros(N+1, dtype=bool)
    candidates = list(range(1, N+1))
    
    while candidates:
        # Filter candidates that don't create 4-AP
        valid_candidates = []
        for x in candidates:
            if not _has_4ap_fast(selected, x):
                # Calculate delta
                d = _delta_3aps(selected, x)
                valid_candidates.append((d, x))
        
        if not valid_candidates:
            break
            
        # Sort by delta descending
        valid_candidates.sort(key=lambda t: t[0], reverse=True)
        
        # Choose from top candidates with some randomness
        # If alpha is low, choose the best. If high, choose randomly from top.
        top_k = max(1, int(len(valid_candidates) * alpha))
        if top_k >= len(valid_candidates):
            top_k = len(valid_candidates)
            
        # Randomly pick from the top_k
        # To be more effective, pick from the top 10 or so
        pick_pool = valid_candidates[:min(top_k, 10)]
        _, x = rng.choice(pick_pool)
        
        selected[x] = True
        candidates.remove(x)
        
    return selected

def _local_search(selected, current_score, deadline, rng):
    """Local search to improve the score."""
    best_selected = selected.copy()
    best_score = current_score
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try swaps: remove x, add y
        # To speed up, we can sample a subset of swaps
        sel_list = [x for x in range(1, N+1) if selected[x]]
        unsel_list = [x for x in range(1, N+1) if not selected[x]]
        
        if not sel_list or not unsel_list:
            break
            
        # Sample some swaps
        n_swaps = min(50, len(sel_list) * len(unsel_list))
        for _ in range(n_swaps):
            x = rng.choice(sel_list)
            y = rng.choice(unsel_list)
            
            # Try swap
            selected[x] = False
            if not _has_4ap_fast(selected, y):
                selected[y] = True
                new_score = current_score - _delta_3aps(selected, x) + _delta_3aps(selected, y)
                # Note: delta calculation above is approximate if interactions exist.
                # For accuracy, we should recalculate or use a better delta.
                # Since N is small, we can just recalculate if the delta suggests improvement.
                if new_score > current_score:
                    # Verify by full count
                    actual_new_score = _count_3aps_fast(selected)
                    if actual_new_score > best_score:
                        best_selected = selected.copy()
                        best_score = actual_new_score
                        current_score = actual_new_score
                        no_improve = 0
                        improved = True
                    else:
                        selected[x] = True
                        selected[y] = False
                else:
                    selected[x] = True
                    selected[y] = False
            else:
                selected[x] = True
                
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return best_selected, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    rng = random.Random(42)
    
    best_selected = None
    best_score = -1
    
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # GRASP construction
        selected = _greedy_construct(rng, alpha=0.3)
        current_score = _count_3aps_fast(selected)
        
        # Local search
        tabu_deadline = min(deadline - 5, time.time() + time_per_restart * 0.8)
        selected, current_score = _local_search(selected, current_score, tabu_deadline, rng)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
    
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
