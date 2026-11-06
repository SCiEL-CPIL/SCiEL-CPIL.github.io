import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    n = 256
    # Initial solution: Suffix [171, 256]
    # Size 86
    S = set(range(171, n + 1))
    best_S = set(S)
    best_size = len(S)
    
    def is_valid(S):
        # Check if S is valid
        # For all a < b <= c in S, a does not divide b+c
        # Optimization: iterate a, then b, then c
        # This is O(N^3) in worst case, but N is small (<= 256)
        # However, we only check during search, so we need it fast.
        # We can precompute conflicts?
        # For now, simple check.
        vals = sorted(S)
        m = len(vals)
        for i in range(m):
            a = vals[i]
            for j in range(i + 1, m):
                b = vals[j]
                # c >= b
                for k in range(j, m):
                    c = vals[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # We need a faster validity check for the search.
    # We can maintain the set of conflicts.
    # Or just check the delta.
    
    def check_add(S, x):
        # Check if adding x to S keeps it valid
        # x must not create a conflict with any pair (b,c) in S
        # and any pair involving x (x, y) must not be a conflict for a in S
        # 1. x as divisor: a=x. Need b,c in S such that x | (b+c).
        # 2. x as summand: b=x or c=x.
        #    If b=x, then a < x. Need a | (x+c).
        #    If c=x, then a < x. Need a | (b+x).
        
        # 1. Check x as divisor
        for b in S:
            if b <= x: continue
            for c in S:
                if c < b: continue
                if (b + c) % x == 0:
                    return False
        
        # 2. Check x as summand
        # b=x, c in S, c >= x
        for c in S:
            if c < x: continue
            # a < x, a in S
            for a in S:
                if a >= x: continue
                if (x + c) % a == 0:
                    return False
        # c=x, b in S, b < x (since b <= c=x, and b != x usually? No, b can be x? No, distinct a,b,c? No, b<=c. If b=c=x, then a < x. a | 2x.)
        # Wait, if b=x, c=x, then a < x. a | 2x.
        # The loop above covers b=x, c=x if c=x is in S.
        # But we are ADDING x. So x is not in S yet.
        # So we check pairs (x, c) where c in S.
        # And (b, x) where b in S.
        # Since b <= c, if c=x, then b <= x.
        # If b=x, c=x, then x is not in S, so this pair doesn't exist in S.
        # So we only check (x, c) with c in S, c >= x.
        # And (b, x) with b in S, b <= x.
        # But (b, x) is same as (x, b) if we swap? No, a < b <= c.
        # If we add x, new pairs are (x, c) and (b, x).
        # If b < x, then (b, x) is valid order.
        # If b > x, then (x, b) is valid order.
        
        for b in S:
            if b > x:
                # Pair (x, b)
                # a < x, a in S
                for a in S:
                    if a >= x: continue
                    if (x + b) % a == 0:
                        return False
            elif b < x:
                # Pair (b, x)
                # a < b, a in S
                for a in S:
                    if a >= b: continue
                    if (b + x) % a == 0:
                        return False
            # if b == x, not in S
        
        return True

    # This check is O(N^2). N=256. 65k ops.
    # We can do ~1000 steps per second.
    # 80s -> 80k steps.
    
    current_S = set(S)
    current_size = len(S)
    
    # Local Search
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Try to add a random element
        candidates = [i for i in range(1, n + 1) if i not in current_S]
        if not candidates:
            break
        
        x = random.choice(candidates)
        if check_add(current_S, x):
            current_S.add(x)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_S = set(current_S)
        else:
            # Try to swap
            # Find a y in current_S to remove such that current_S - {y} + {x} is valid
            # This is expensive.
            # Just skip for now.
            pass
            
    return list(best_S)
