import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    # Precompute divisibility table for speed
    # divs[a] = set of numbers x in [1, 512] such that a divides x
    # We need b+c where b,c in [1,256], so b+c in [3, 512]
    # Actually a can be up to 256, b+c up to 512
    # Let's just use a simple check but optimize the inner loops
    
    # Proven valid tail: [171, 256]
    # For a >= 171, b > a >= 171, c >= b > 171
    # b+c > 342.
    # a divides b+c implies b+c >= 2a (since b+c > a).
    # Max b+c = 512.
    # If a=171, 2a=342, 3a=513>512. So only possible multiple is 2a.
    # b+c = 2a = 342.
    # b > 171, c >= b => b >= 172, c >= 172 => b+c >= 344 > 342.
    # So no solution.
    # This holds for all a in [171, 256].
    base_tail = list(range(171, 257))
    
    def can_add(S, x):
        """Check if adding x to set S keeps validity. S is a set of ints."""
        if x in S:
            return True
        
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: x is smallest (a=x)
        # Need b,c in S with x < b <= c and x | (b+c)
        # We can optimize by checking sums
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, m):
                c = S_list[j]
                # c >= b is guaranteed since sorted
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is middle (b=x)
        # Need a in S with a < x, c in S with c >= x, a | (x+c)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            for j in range(m):
                c = S_list[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is largest (c=x)
        # Need a,b in S with a < b <= x, a | (b+x)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True

    def try_greedy(S_initial, order):
        """Try greedy addition in given order."""
        S = set(S_initial)
        for x in order:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    best_set = set(base_tail)
    best_size = len(base_tail)
    
    candidates_to_add = list(range(1, 171))

    # Strategy 1: Descending order
    order_desc = list(range(170, 0, -1))
    S1 = try_greedy(base_tail, order_desc)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    if time.time() > deadline:
        return sorted(best_set)

    # Strategy 2: Ascending order
    order_asc = list(range(1, 171))
    S2 = try_greedy(base_tail, order_asc)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    if time.time() > deadline:
        return sorted(best_set)

    # Strategy 3: Conflict-priority greedy
    def count_conflicts(x, current_set):
        """Count conflicts for adding x to current_set."""
        S_list = sorted(current_set)
        m = len(S_list)
        count = 0
        
        # Case 1: x is smallest
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    count += 1
                    break
        
        # Case 2: x is middle
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            for j in range(m):
                c = S_list[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    count += 1
                    break
        
        # Case 3: x is largest
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    count += 1
                    break
        
        return count

    def greedy_with_conflict_priority(S_initial, candidates, deadline):
        S = set(S_initial)
        remaining = [x for x in candidates if x not in S]
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_candidate = None
            best_conflicts = float('inf')
            
            for x in remaining:
                if x in S:
                    continue
                conflicts = count_conflicts(x, S)
                if conflicts < best_conflicts or (conflicts == best_conflicts and (best_candidate is None or x < best_candidate)):
                    best_conflicts = conflicts
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            if can_add(S, best_candidate):
                S.add(best_candidate)
            remaining.remove(best_candidate)
        
        return S

    S3 = greedy_with_conflict_priority(base_tail, candidates_to_add, deadline)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(S3)
    
    if time.time() > deadline:
        return sorted(best_set)

    # Strategy 4: Try lower tail starts
    def is_valid_set(S):
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    for tail_start in range(170, 140, -1):
        if time.time() > deadline:
            break
        tail = list(range(tail_start, 257))
        if is_valid_set(tail):
            order_below = list(range(tail_start - 1, 0, -1))
            S_new = greedy_with_conflict_priority(tail, order_below, deadline)
            if len(S_new) > best_size:
                best_set = S_new
                best_size = len(S_new)
            break

    if time.time() > deadline:
        return sorted(best_set)

    # Strategy 5: Random restarts with greedy
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 2:
            break
        order = candidates_to_add[:]
        rng.shuffle(order)
        S_t = try_greedy(base_tail, order)
        if len(S_t) > best_size:
            best_set = S_t
            best_size = len(S_t)

    # Strategy 6: Local search / Swap optimization on best_set
    # Try to swap elements in best_set with candidates to increase size
    # This is complex, so we'll stick to greedy variations if time permits
    
    # One more pass: try to add any remaining candidates to best_set
    S_final = set(best_set)
    remaining_candidates = [x for x in range(1, 257) if x not in S_final]
    # Sort by value, try adding
    for x in sorted(remaining_candidates):
        if time.time() > deadline - 1:
            break
        if can_add(S_final, x):
            S_final.add(x)
    
    if len(S_final) > best_size:
        best_set = S_final
        best_size = len(S_final)

    return sorted(best_set)
