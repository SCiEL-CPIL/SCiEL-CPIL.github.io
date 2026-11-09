"""
Snake-in-the-box in Q_9: construct longest induced path.

Approach: Iterated Local Search (ILS) with aggressive restarts.
1. Fast greedy initialization with random tie-breaking.
2. Local search: try to extend, pivot, or retract.
3. When stuck, perturb the path (random walk backward) and restart local search.
4. Keep track of the best path found.
"""

import random
import time

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_extension(candidate, path, used):
    """Check if candidate can be appended to path to keep it induced."""
    if candidate in used:
        return False
    if (candidate ^ path[-1]).bit_count() != 1:
        return False
    for p in path[:-1]:
        if (candidate ^ p).bit_count() == 1:
            return False
    return True

def _greedy_init(start=0):
    """Quick greedy initialization."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        if not candidates:
            break
        next_v = random.choice(candidates)
        path.append(next_v)
        used.add(next_v)
    return path

def _local_search(path, used, time_limit):
    """Perform local search on the path until time runs out or no moves available."""
    best_path = list(path)
    best_len = len(path)
    
    while time.time() < time_limit:
        moved = False
        
        # Try to extend
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                path.append(v)
                used.add(v)
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                moved = True
                break
        
        if moved:
            continue
        
        # Try to pivot: replace last vertex with another valid neighbor of second-to-last
        if len(path) >= 2:
            prev = path[-2]
            last = path[-1]
            candidates = []
            for b in range(N):
                v = prev ^ (1 << b)
                if v in used or v == last:
                    continue
                valid = True
                for p in path[:-2]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if candidates:
                new_last = random.choice(candidates)
                used.remove(last)
                used.add(new_last)
                path[-1] = new_last
                moved = True
                continue
        
        # Try to retract: remove last vertex
        if len(path) > 2:
            # Accept retraction with some probability to escape local optima
            if random.random() < 0.1:
                last = path.pop()
                used.remove(last)
                moved = True
                continue
        
        # No move available, break to allow perturbation
        break
    
    return best_path, best_len

def construct():
    """
    Construct the longest snake in Q_9 using Iterated Local Search.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    random.seed(42)
    
    best_path = []
    best_len = 0
    
    # Multiple restarts with perturbations
    num_restarts = 50
    restart_time = (time_limit - start_time) / num_restarts
    
    for _ in range(num_restarts):
        if time.time() - start_time >= time_limit:
            break
        
        # Initialize
        current_path = _greedy_init(0)
        current_used = set(current_path)
        
        # Local search with a time budget for this restart
        restart_deadline = time.time() + min(restart_time, time_limit - (time.time() - start_time))
        best_path_ls, best_len_ls = _local_search(current_path, current_used, restart_deadline)
        
        if best_len_ls > best_len:
            best_path = best_path_ls
            best_len = best_len_ls
        
        # Perturb: truncate the best path found and try again from there
        if best_len > 10:
            truncate_point = max(5, best_len - random.randint(5, 20))
            current_path = list(best_path[:truncate_point])
            current_used = set(current_path)
            
            # Continue local search from truncated state
            restart_deadline2 = time.time() + min(restart_time * 0.5, time_limit - (time.time() - start_time))
            best_path_ls2, best_len_ls2 = _local_search(current_path, current_used, restart_deadline2)
            
            if best_len_ls2 > best_len:
                best_path = best_path_ls2
                best_len = best_len_ls2
    
    return best_path

# EVOLVE-BLOCK-END
