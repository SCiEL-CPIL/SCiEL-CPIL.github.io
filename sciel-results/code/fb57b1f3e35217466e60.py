import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Base solution: upper tail from 171 to 256
    s = set(range(171, 257))
    best_set = set(s)
    best_size = len(s)

    def can_add(x, current_set):
        """Check if x can be added to current_set without violating the property."""
        elems = sorted(current_set)
        
        # Case 1: x is 'a' (smallest in triple)
        # Need: x does NOT divide b+c for all b,c in current_set with x<b<=c
        for i, b in enumerate(elems):
            if b <= x:
                continue
            for c in elems[i:]:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle in triple)
        # Need: a does NOT divide x+c for all a in current_set with a<x, c in current_set with x<=c
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest in triple)
        # Need: a does NOT divide b+x for all a,b in current_set with a<b<=x
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def full_check(S):
        """Verify the entire set satisfies the property."""
        elems = sorted(S)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Precompute conflict info: for each candidate x in [1,170],
    # determine which base elements would conflict
    # Use a smarter greedy: sort candidates by how "safe" they are (fewer conflicts)
    
    # Compute a score for each candidate: number of conflicts with base set
    candidates = list(range(170, 0, -1))
    
    # Try multiple strategies
    strategies = [
        ('descending', list(range(170, 0, -1))),
        ('ascending', list(range(1, 171))),
        ('prime_first', [x for x in range(170, 0, -1) if (all(x % p != 0 for p in range(2, int(x**0.5)+1)) or x == 1)] + [x for x in range(170, 0, -1) if not (all(x % p != 0 for p in range(2, int(x**0.5)+1)) or x == 1)]),
    ]
    
    for trial in range(30):
        if time.time() > deadline - 2:
            break
        
        s_trial = set(range(171, 257))
        
        # Choose strategy based on trial
        if trial < 5:
            order = list(range(170, 0, -1))  # descending
        elif trial < 10:
            order = list(range(1, 171))  # ascending
        elif trial < 15:
            # Try only numbers coprime to many base elements
            order = sorted(range(1, 171), key=lambda x: sum(1 for y in range(171, 257) if (x + y) % x == 0 or (y + y) % x == 0))
        elif trial < 20:
            # Random shuffle
            order = list(range(170, 0, -1))
            random.shuffle(order)
        else:
            # Try numbers with few divisors in range
            order = sorted(range(1, 171), key=lambda x: sum(1 for d in range(2, x+1) if x % d == 0))
        
        for x in order:
            if time.time() > deadline - 1:
                break
            if can_add(x, s_trial):
                s_trial.add(x)
        
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Local search: try to improve by swapping elements
    for _ in range(10):
        if time.time() > deadline - 1:
            break
        
        s_trial = set(best_set)
        improved = False
        
        for rem in list(s_trial):
            if time.time() > deadline - 1:
                break
            s_trial.remove(rem)
            for add in range(1, 257):
                if add in s_trial or add == rem:
                    continue
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                    break
            s_trial.add(rem)
            if improved:
                break

    # Final validation
    if full_check(best_set):
        return list(best_set)
    
    # Fallback to known valid solution
    return list(range(171, 257))
