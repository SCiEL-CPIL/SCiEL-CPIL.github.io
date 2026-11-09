import time
import random

def construct():
    n = 128
    start_time = time.time()
    deadline = 85.0
    
    # Precompute hyperedges: for each triple (a,b,c) with a<b<c,
    # check if a+b, a+c, b+c are all in [1,n].
    # A hyperedge means we cannot have all of {a,b,c,a+b,a+c,b+c} in our set.
    # Actually, the constraint is: no three DISTINCT members a,b,c such that
    # a+b, a+c, b+c are also selected.
    # So the forbidden configuration is: a,b,c in S AND a+b in S AND a+c in S AND b+c in S.
    # Note: a,b,c must be distinct, and the sums may coincide with each other or with a,b,c.
    
    # Bit-parallel approach: represent S as a bitmask.
    # For checking if we can add x to S, we need to verify no forbidden config is created.
    
    # Let's precompute for each pair (a,b) with a<b, the "partner" c values that would
    # create a forbidden config if a,b,c,x are all in S.
    
    # Actually, let me think differently. The forbidden pattern involves 6 values:
    # a, b, c, a+b, a+c, b+c where a,b,c are distinct elements of S.
    # When we add x, x can be one of a,b,c or one of the sums.
    
    # Case 1: x is one of {a,b,c}. WLOG x=a. Then we need b,c in S such that
    #   a+b in S, a+c in S, b+c in S (or b+c == a, but b+c > a since b,c >= 1 and distinct from a)
    #   Wait, b+c could equal a if b+c = a. But since all are positive and distinct, 
    #   b+c = a is possible if a > b and a > c.
    
    # Case 2: x is one of the sums. WLOG x = a+b. Then we need a,b,c in S (distinct) such that
    #   a+c in S and b+c in S.
    
    # For efficiency, let's use bitmasks.
    # S_mask: bit i is set if i+1 is in S (0-indexed, so bit 0 = value 1)
    
    def can_add_mask(S_mask, x):
        """Check if adding x (1-indexed) to S_mask is valid."""
        x_idx = x - 1  # 0-indexed
        S_mask_with_x = S_mask | (1 << x_idx)
        
        # Case 1: x is one of a,b,c. WLOG a=x.
        # Need b, c in S (distinct, not x) such that x+b in S, x+c in S, b+c in S.
        # b+c could equal x, but let's handle that.
        for b in range(1, n + 1):
            if b == x:
                continue
            if not (S_mask & (1 << (b - 1))):
                continue
            xb = x + b
            if xb > n:
                continue
            if not (S_mask & (1 << (xb - 1))):
                continue
            # Now need c in S, c != x, c != b, such that x+c in S and b+c in S
            for c in range(1, n + 1):
                if c == x or c == b:
                    continue
                if not (S_mask & (1 << (c - 1))):
                    continue
                xc = x + c
                if xc > n:
                    continue
                if not (S_mask & (1 << (xc - 1))):
                    continue
                bc = b + c
                if bc > n:
                    continue
                # bc must be in S OR bc == x (since the six values need not be distinct)
                if bc == x:
                    return False
                if not (S_mask & (1 << (bc - 1))):
                    continue
                return False
        
        # Case 2: x is one of the sums. WLOG x = a+b.
        # Need a, b, c in S (all distinct from each other, and distinct from x? No, just a,b,c distinct)
        # such that a+c in S and b+c in S.
        for a in range(1, n + 1):
            b = x - a
            if b <= 0 or b >= x:  # b > 0 and b != a (since a+b=x, if a==b then 2a=x)
                continue
            if a == b:
                continue
            if not (S_mask & (1 << (a - 1))):
                continue
            if not (S_mask & (1 << (b - 1))):
                continue
            # Now need c in S, c != a, c != b, such that a+c in S and b+c in S
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if not (S_mask & (1 << (c - 1))):
                    continue
                ac = a + c
                if ac > n:
                    continue
                if not (S_mask & (1 << (ac - 1))):
                    continue
                bc = b + c
                if bc > n:
                    continue
                if not (S_mask & (1 << (bc - 1))):
                    continue
                return False
        
        return True
    
    # Greedy with random ordering using bitmask
    best_mask = 0
    best_size = 0
    
    # Start with upper half as initial seed
    S0_mask = 0
    for i in range(n // 2 + 1, n + 1):
        S0_mask |= (1 << (i - 1))
    best_mask = S0_mask
    best_size = n // 2
    
    candidates = list(range(1, n // 2 + 1))
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        S_mask = S0_mask
        current_size = n // 2
        
        for x in candidates:
            if time.time() > deadline:
                break
            if S_mask & (1 << (x - 1)):
                continue
            if can_add_mask(S_mask, x):
                S_mask |= (1 << (x - 1))
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # Local search: remove random element, try to add others
    S_mask = best_mask
    S_list = [i for i in range(1, n + 1) if S_mask & (1 << (i - 1))]
    
    while time.time() < deadline:
        if not S_list:
            break
        
        y = random.choice(S_list)
        S_mask &= ~(1 << (y - 1))
        S_list.remove(y)
        
        improved = False
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if S_mask & (1 << (x - 1)):
                continue
            if can_add_mask(S_mask, x):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
                improved = True
        
        if not improved:
            # Restore
            S_mask |= (1 << (y - 1))
            S_list.append(y)
            break
    
    # Extract result
    result = [i for i in range(1, n + 1) if best_mask & (1 << (i - 1))]
    return result
