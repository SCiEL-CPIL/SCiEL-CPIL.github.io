import time
import random
import math

_START = time.time()
_DEADLINE = _START + 115.0

def _score(u):
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    # Compute unique sums and differences
    # For small n, O(n^2) is fast
    u_set = set(u)
    sums = set()
    diffs = set()
    for a in u:
        for b in u:
            sums.add(a + b)
            diffs.add(a - b)
            
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _construct_golomb(n):
    """
    Construct a Golomb ruler of length n using a simple greedy algorithm.
    This ensures |U-U| is maximized (all differences distinct).
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Greedy construction
    u = [0]
    for i in range(1, n):
        best_pos = -1
        best_diff_count = -1
        # Try positions from 1 upwards
        # Upper bound heuristic: i^2 + i
        max_pos = (i+1)**2 + 10
        for pos in range(1, max_pos + 1):
            # Check if pos creates duplicate differences
            is_golomb = True
            for existing in u:
                d = abs(pos - existing)
                # Check if this difference already exists in the set of differences
                # We can maintain a set of differences, but for small n, just check
                for other in u:
                    if other != existing:
                        if abs(pos - other) == d or abs(existing - other) == d:
                            # Wait, we need to check if any pair in u has difference d,
                            # or if pos-existing matches any existing pair difference.
                            pass
            # Simpler check:
            diffs = set()
            valid = True
            for a in u:
                for b in u:
                    if a != b:
                        diffs.add(abs(a-b))
            # Now check new element
            for a in u:
                d = abs(pos - a)
                if d in diffs:
                    valid = False
                    break
            if valid:
                # Accept first valid position? Or the one that minimizes span?
                # Greedy: take the smallest valid position.
                best_pos = pos
                break
        
        if best_pos == -1:
            # Fallback: just append i
            u.append(i)
        else:
            u.append(best_pos)
            
    return sorted(u)

def _local_search(u, timeout=2.0):
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
        
    current_score = _score(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    # We will try to improve by changing one element at a time.
    # Search space for a new value: [0, max_val * 1.5]
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 2.0:
        improved = False
        # Try each element
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 2.0:
                break
            old_val = u[i]
            # Try a range of new values
            # Small perturbations and some larger jumps
            search_range = max(5, max_val // 2)
            # Sample candidates
            candidates = set()
            # Local neighborhood
            for delta in range(-search_range, search_range + 1, max(1, search_range // 20)):
                new_val = old_val + delta
                if new_val >= 0:
                    candidates.add(new_val)
            # Random jumps
            for _ in range(5):
                candidates.add(random.randint(0, max_val + search_range))
                
            for new_val in candidates:
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score(new_u)
                if s > current_score:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Random restart or mutation if stuck
            if random.random() < 0.1:
                # Perturb a random element significantly
                idx = random.randint(0, n-1)
                u[idx] = random.randint(0, max_val + 10)
                u = sorted(set(u))
                if len(u) < n:
                    # If collision, try to fix or accept lower n? 
                    # Better to revert if n changes, as we are optimizing for fixed n in this loop
                    u = best_u[:]
                else:
                    current_score = _score(u)

    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Focus on small n where the score potential is highest
    # The score term log(R)/log(2M+1) decays as M increases.
    # Small n allows small M.
    
    for n in range(2, 16):
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Allocate time per n
        # Smaller n should be faster to optimize, but we can give them equal time
        # or slightly less if they converge quickly.
        time_budget = 3.0
        
        # Multiple restarts
        # 1. Start from Golomb ruler
        # 2. Start from random
        restarts = 5
        
        for r in range(restarts):
            if time.time() > _DEADLINE - 5.0:
                break
                
            if r == 0:
                # Golomb ruler initialization
                u_init = _construct_golomb(n)
            else:
                # Random initialization
                # Span heuristic: n^2
                max_span = n * n
                u_init = [0]
                for _ in range(n - 1):
                    u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
                if len(u_init) < n:
                    # Ensure size n
                    while len(u_init) < n:
                        u_init.append(random.randint(1, max_span))
                        u_init = sorted(set(u_init))
            
            u_opt, s_opt = _local_search(u_init, timeout=time_budget)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Final check
    return best_u
