import time
import random

N = 512
TIME_LIMIT = 78  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    # Precompute: for each element x, list of tuples of 3 other elements
    # such that {x} + those 3 form a 4-AP
    aps_containing = {x: [] for x in range(1, N + 1)}
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap = (a, a + d, a + 2*d, a + 3*d)
            for i in range(4):
                x = ap[i]
                others = tuple(ap[j] for j in range(4) if j != i)
                aps_containing[x].append(others)
    
    def can_add(x, current_set):
        """Check if adding x to current_set creates a 4-term AP"""
        for others in aps_containing[x]:
            if all(o in current_set for o in others):
                return False
        return True
    
    def greedy_randomized():
        """Greedy with random ordering"""
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        current_set = set()
        for x in elements:
            if can_add(x, current_set):
                current_set.add(x)
        return current_set
    
    def local_search(current_set, max_time):
        """Improve by adding elements or 1-for-1 swaps with greedy refill"""
        current_set = set(current_set)
        local_start = time.time()
        
        improved = True
        while improved and (time.time() - local_start) < max_time:
            improved = False
            
            # Phase 1: Try adding any element
            for x in range(1, N + 1):
                if x not in current_set and can_add(x, current_set):
                    current_set.add(x)
                    improved = True
                    break
            
            # Phase 2: Try 1-for-1 swap with greedy refill
            if not improved:
                for remove_x in list(current_set):
                    if (time.time() - local_start) > max_time:
                        break
                    trial = current_set - {remove_x}
                    for add_x in range(1, N + 1):
                        if add_x not in trial and can_add(add_x, trial):
                            trial.add(add_x)
                    if len(trial) > len(current_set):
                        current_set = trial
                        improved = True
                        break
        return current_set
    
    best_set = set()
    best_size = 0
    
    # Phase A: Many fast randomized greedy + quick local search
    while time_left() > 10:
        s = greedy_randomized()
        s = local_search(s, max_time=0.3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Phase B: Deeper local search on best
    if time_left() > 3:
        s = local_search(best_set, max_time=time_left() - 1)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Phase C: Final single-addition polish
    if time_left() > 0.5:
        for x in range(1, N + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                best_size = len(best_set)
    
    return sorted(best_set)
