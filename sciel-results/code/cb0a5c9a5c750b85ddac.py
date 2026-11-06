"""Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
import time
import sys
import itertools

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

def _search(M, target_diffs, deadline):
    """
    Depth-first search for a Sidon set in [0, M] covering target_diffs.
    Uses forward checking and strong heuristics.
    """
    initial_set = [0]
    initial_diffs = set()
    remaining = set(target_diffs)
    
    def dfs(current_set, used_diffs, remaining):
        if not remaining:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
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
                    
                    # Count how many remaining diffs this x would cover
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
        
        for x, cov in candidate_info:
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
    """Greedy construction: pick smallest x that covers most remaining diffs without conflict."""
    a = [0]
    used = set()
    required = set(range(1, 31))
    
    for _ in range(31):  # at most 32 elements total
        if not required:
            break
        best_x = None
        best_cov = -1
        for x in range(a[-1] + 1, 200):
            new_diffs = set()
            ok = True
            for v in a:
                d = x - v
                if d in used:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                cov = len(new_diffs & required)
                if cov > best_cov:
                    best_cov = cov
                    best_x = x
        if best_x is None:
            break
        a.append(best_x)
        for v in a[:-1]:
            d = best_x - v
            used.add(d)
        required -= set(best_x - v for v in a[:-1])
    
    return a if not required else None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # First try greedy to get a quick valid solution
    greedy_result = _greedy_construct()
    if greedy_result is not None:
        greedy_max = max(greedy_result)
    else:
        greedy_max = 200
    
    # Try DFS from smaller M values up to greedy_max
    for M in range(55, min(greedy_max + 1, 150)):
        if time.time() > deadline:
            break
            
        result = _search(M, target_diffs, deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    # Fallback to greedy
    if greedy_result is not None:
        return greedy_result
    
    # Last resort
    return [0, 1, 4, 10, 18, 22, 26, 30, 35]
