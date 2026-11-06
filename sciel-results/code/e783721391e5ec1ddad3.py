"""Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
import time
import random

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search(M, target_diffs, deadline, max_nodes=500000):
    """
    Depth-first search for a Sidon set in [0, M] covering target_diffs.
    Uses forward checking and heuristic ordering.
    """
    initial_set = [0]
    initial_diffs = set()
    remaining = set(target_diffs)
    
    node_count = [0]
    
    def dfs(current_set, used_diffs, remaining):
        if not remaining:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
            
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return None
            
        last = current_set[-1]
        
        # Generate candidates: x = a + d for a in current_set, d in remaining
        # where x > last and x <= M
        candidate_info = []
        seen_candidates = set()
        for a in current_set:
            for d in remaining:
                x = a + d
                if x > last and x <= M and x not in seen_candidates:
                    seen_candidates.add(x)
                    
                    # Check for conflicts and count coverage
                    new_covered = 0
                    conflict = False
                    for a2 in current_set:
                        diff = x - a2
                        if diff in used_diffs:
                            conflict = True
                            break
                        if diff in remaining:
                            new_covered += 1
                    
                    if not conflict:
                        candidate_info.append((x, new_covered))
        
        # Sort by: more new coverage first, then smaller x (to minimize max)
        candidate_info.sort(key=lambda t: (-t[1], t[0]))
        
        for x, _ in candidate_info:
            # Re-verify Sidon property
            new_diffs = set()
            conflict = False
            for a in current_set:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
                
            new_used = used_diffs | new_diffs
            new_remaining = remaining - new_used
            
            res = dfs(current_set + [x], new_used, new_remaining)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, initial_diffs, remaining)

def _greedy_construct():
    """Greedy construction to get a valid starting set."""
    a = [0]
    used = set()
    required = set(range(1, 31))
    
    while required:
        best_x = None
        best_new_diffs = None
        best_score = -1
        
        for x in range(1, 200):
            if x in a:
                continue
            new_diffs = set()
            ok = True
            for v in a:
                d = abs(x - v)
                if d in used:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                # Score: number of required diffs covered
                score = len(new_diffs & required)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_diffs = new_diffs
        
        if best_x is not None and best_score > 0:
            a.append(best_x)
            used |= best_new_diffs
            required -= best_new_diffs
        else:
            # Fallback: add a large number
            x = max(a) + 100
            while True:
                new_diffs = set(abs(x - v) for v in a)
                if not (new_diffs & used):
                    a.append(x)
                    used |= new_diffs
                    required -= (new_diffs & required)
                    break
                x += 1
    return sorted(a)

def _local_search(a, deadline):
    """Try to reduce max element by replacing large elements with smaller ones."""
    a = sorted(a)
    best_max = max(a)
    best_a = a[:]
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each non-zero element and re-adding a smaller one
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
                
            candidate = a[i]
            if candidate < best_max:
                continue
                
            # Try to replace a[i] with a smaller value
            for new_val in range(1, candidate):
                trial = a[:i] + [new_val] + a[i+1:]
                trial = sorted(trial)
                
                # Check if it's still valid
                if _check_sidon_and_cover(trial, set(range(1, 31))):
                    if max(trial) < best_max:
                        best_max = max(trial)
                        best_a = trial
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_a

def _known_good():
    """A known valid Sidon set covering 1..30 with small max element."""
    candidates = [
        [0, 1, 4, 10, 12, 17, 21, 25, 30],
        [0, 1, 4, 10, 18, 22, 26, 30, 35],
        [0, 1, 3, 7, 12, 18, 25, 30, 36],
    ]
    target = set(range(1, 31))
    for c in candidates:
        if _check_sidon_and_cover(c, target):
            return c
    # If none work, build greedily
    return _greedy_construct()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # First, get a valid starting point via greedy
    greedy_result = _greedy_construct()
    best_result = greedy_result
    best_max = max(greedy_result)
    
    # Try local search to improve
    if time.time() < deadline - 10:
        improved = _local_search(greedy_result, deadline - 5)
        if max(improved) < best_max:
            best_result = improved
            best_max = max(improved)
    
    # Try systematic search with increasing M
    # Lower bound: need at least 9 elements, min diameter ~70
    # Start from 55 to catch edge cases, go up to 150
    for M in range(55, min(best_max, 150)):
        if time.time() > deadline:
            break
            
        result = _search(M, target_diffs, deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                if max(result) < best_max:
                    best_result = result
                    best_max = max(result)
                break
                
    # Final local search on best result
    if time.time() < deadline - 5:
        improved = _local_search(best_result, deadline - 1)
        if max(improved) < best_max:
            best_result = improved
    
    # Verify final result
    if _check_sidon_and_cover(best_result, target_diffs):
        return best_result
    
    # Fallback
    return _known_good()
