import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy Removal from high-density starts
    # Start with sets of large numbers
    start_vals = [128, 129, 130, 120, 110, 100, 140, 150, 160]
    
    for start_val in start_vals:
        if time.time() > deadline - 8:
            break
        
        S = set(range(start_val, n + 1))
        
        # Resolve conflicts by removing elements with highest conflict degree
        S = _resolve_conflicts(S, n, deadline - 8)
        
        # Try to add back any excluded elements
        if time.time() < deadline - 15:
            S = _try_add_back(S, n, deadline - 15)
        
        # Local search: try swaps to improve
        if time.time() < deadline - 8:
            S = _local_search(S, n, deadline - 8)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Strategy 2: Original Greedy (Largest to Smallest)
    if time.time() < deadline - 5:
        S_greedy = set()
        for x in range(n, 0, -1):
            if time.time() > deadline:
                break
            if _can_add(x, S_greedy, n):
                S_greedy.add(x)
        
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_set = set(S_greedy)

    if not best_set:
        return [n]
    
    return sorted(best_set)

def _can_add(x, S, n):
    """Check if x can be added to S without violating the property."""
    S_list = sorted(S)
    S_set = S
    
    # Case 1: x is the smallest (a=x), need b > x, c >= b in S, x | (b+c)
    for b in S_list:
        if b <= x:
            continue
        # For this b, check if any c >= b in S has (b+c) % x == 0
        # i.e., c ≡ -b (mod x) and c >= b
        target = (-b) % x
        # Find c in S_list where c >= b and c % x == target
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b or c
    for a in S_list:
        if a >= x:
            continue
        # x as b: need c in S with x <= c and a | (x+c)
        # i.e., c ≡ -x (mod a) and c >= x
        target_c = (-x) % a
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
        # x as c: need b in S with a < b <= x and a | (b+x)
        # i.e., b ≡ -x (mod a) and a < b <= x
        target_b = (-x) % a
        for b in S_list:
            if b <= a or b > x:
                continue
            if (b + x) % a == 0:
                return False
    
    return True

def _resolve_conflicts(S, n, deadline):
    """Remove elements from S to resolve all conflicts."""
    S = set(S)
    while True:
        if time.time() > deadline:
            break
        
        violations = []
        S_list = sorted(S)
        # Find all violations
        for a in S_list:
            for b in S_list:
                if b <= a:
                    continue
                for c in S_list:
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        violations.append((a, b, c))
        
        if not violations:
            break
        
        # Count violations per element, weighted: removing small a is costly
        # because it participates as 'a' in many triples.
        # Score = num_violations / (element_value / 10)  -- prefer removing larger elements
        # Actually: we want to remove the element whose removal fixes the most violations
        # relative to its "importance". Smaller numbers are more important (they appear
        # as 'a' in more potential conflicts). So penalize removing small numbers.
        degrees = {}
        for a, b, c in violations:
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
            degrees[c] = degrees.get(c, 0) + 1
        
        # Pick element to remove: maximize (degree / element_value) to prefer
        # removing large elements that have high conflict count
        best_score = -1
        x_to_remove = None
        for x, d in degrees.items():
            score = d / (x + 1)  # prefer larger x with high degree
            if score > best_score:
                best_score = score
                x_to_remove = x
        
        if x_to_remove is None:
            x_to_remove = min(degrees.keys())
        
        S.remove(x_to_remove)
        
    return S

def _try_add_back(S, n, deadline):
    """Try to add any element from 1..n back into S if possible."""
    S = set(S)
    # Larger values are easier to add.
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if x in S:
            continue
        if _can_add(x, S, n):
            S.add(x)
    return S

def _local_search(S, n, deadline):
    """Try swaps: remove one element, add another, if it increases size or maintains."""
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(S)
        # Try removing each element and see if we can add more than one back
        for i, rem in enumerate(S_list):
            if time.time() > deadline:
                break
            S_trial = set(S)
            S_trial.discard(rem)
            # Try to add back elements not in S_trial
            added = 0
            for x in range(n, 0, -1):
                if time.time() > deadline:
                    break
                if x in S_trial:
                    continue
                if x == rem:
                    continue
                if _can_add(x, S_trial, n):
                    S_trial.add(x)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                S = S_trial
                improved = True
                break
    return S
