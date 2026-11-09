import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_mask = (1 << n) - 1
    
    # Precompute conflict masks: for each element x, which elements y conflict with x
    # x and y conflict if adding both creates a forbidden triple
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c are all in the set
    # So x conflicts with y if there exists z such that {x,y,z} forms a bad triple
    # OR if x+y is in the set and some z makes it bad
    
    # Better approach: precompute for each pair (x,y) whether they can coexist
    # Actually, let's precompute the "conflict" relationship:
    # Two elements x, y conflict if there exists z such that all of x, y, z, x+y, x+z, y+z are in domain [1..128]
    # and if we have x,y,z in our set, then x+y, x+z, y+z must NOT all be in our set.
    
    # For greedy MIS: process vertices in random order, add vertex if it doesn't conflict with any already-added vertex.
    # Conflict: x and y conflict if there exists z in [1..128] such that:
    #   x+y <= 128, x+z <= 128, y+z <= 128
    #   and if x,y,z are all selected, then x+y, x+z, y+z would need to be checked.
    # Actually the constraint is: NO three distinct a,b,c in S such that a+b, a+c, b+c are ALL in S.
    # So x and y don't directly conflict; they conflict via a third element z.
    
    # For greedy: when considering adding x, check if it creates a bad triple with any two already-selected elements.
    # This is O(|S|^2) per addition, which is too slow for 128 elements.
    
    # Alternative: use bit-parallel checking.
    # For a given set S (as bitmask), and candidate x, check validity:
    # Case 1: x is the "third vertex" c. Need a,b in S, a!=b, a!=x, b!=x, such that a+x in S, b+x in S, a+b in S.
    # Case 2: x is a sum, say x = a+b. Need c in S, c!=a, c!=b, such that a+c in S, b+c in S.
    
    # Let's precompute for each x the list of (a,b) pairs where x could be c=a+b or x could be a vertex.
    
    # Precompute: for each x in 1..128, store:
    #   sums_of_x: list of (a,b) with a<b, a+b=x, both in 1..128
    #   pairs_with_x: for case where x is a vertex, we need a,b such that a+x in range, b+x in range, a+b in range
    #   Actually for case 1 (x is c): a,b in S, a+x in S, b+x in S, a+b in S
    #   We can precompute for each x the set of (a,b) pairs where a+x<=128, b+x<=128, a+b<=128
    
    # Let me just implement a fast can_add function using bitmasks.
    
    def can_add(mask, x):
        """Check if x can be added to set represented by mask. x is 1-indexed."""
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is the third vertex c.
        # Find all a in S such that a+x in S (and a != x, a+x != x which is always true if a>0)
        # a ranges from 1 to n-x
        # For each such a, check if there exists b in S, b != a, b != x, such that b+x in S and a+b in S (or a+b == x)
        
        # Build mask of valid a's: a in S and a+x in S
        # a in 1..n-x, so bits 0..n-x-1
        # mask_a = (mask & (mask >> x)) & ((1 << (n - x)) - 1)
        # This gives bits where a is in S and a+x is in S
        
        if x <= n:
            mask_a = (mask & (mask >> x)) & ((1 << (n - x)) - 1)
            # Remove bit x-1 if x-1 < n-x, i.e., 2x <= n+1... actually a != x means bit (x-1) should be cleared
            if x - 1 < n - x:
                mask_a &= ~(1 << (x - 1))
            
            # Now for each a in mask_a, we need to check if there's a b in mask_a (b != a) such that a+b in S or a+b == x
            # If mask_a has fewer than 2 bits, no conflict in Case 1
            if mask_a & (mask_a - 1):  # at least 2 bits set
                # For each pair (a,b) in mask_a, check if a+b in S or a+b == x
                # This is expensive. Alternative: for each a, check if (a+b) is in S for any b in mask_a \ {a}
                # Let's iterate over set bits in mask_a
                a_list = []
                m = mask_a
                while m:
                    lb = m & (-m)
                    a_list.append(lb.bit_length())  # 1-indexed position
                    m ^= lb
                
                if len(a_list) < 2:
                    pass  # no conflict
                else:
                    for i in range(len(a_list)):
                        a = a_list[i]
                        # For this a, check all b in a_list where b != a
                        # b must satisfy: b+x in S (already guaranteed by mask_a), and a+b in S or a+b == x
                        for j in range(len(a_list)):
                            if i == j:
                                continue
                            b = a_list[j]
                            s = a + b
                            if s == x:
                                return False
                            if s <= n and (mask >> (s - 1)) & 1:
                                return False
        
        # Case 2: x = a + b for some a < b, a+b = x, a,b in S
        # Need c in S, c != a, c != b, such that a+c in S and b+c in S
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                # c must be in S, c != a, c != b, a+c in S, b+c in S
                # a+c in S: c in 1..n-a, bit (c-1) in mask, bit (a+c-1) in mask
                # So c-1 must be in mask & (mask >> a)
                # Similarly c-1 in mask & (mask >> b)
                C_mask = (mask & (mask >> a)) & (mask & (mask >> b))
                # Clear bits for a-1 and b-1
                C_mask &= ~(1 << (a - 1))
                C_mask &= ~(1 << (b - 1))
                # Also c should not make a+c or b+c exceed n, but that's handled by the shift
                if C_mask != 0:
                    return False
        
        return True
    
    def greedy_mis(order):
        """Greedy maximal independent set with given vertex order."""
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask
    
    def local_search(mask, time_limit):
        """Local search: try to add/remove elements to improve."""
        best = mask
        best_size = bin(mask).count('1')
        deadline = time.time() + time_limit
        while time.time() < deadline:
            # Try adding a random missing element
            missing = [x for x in range(1, n + 1) if not (mask >> (x - 1)) & 1]
            if missing:
                x = random.choice(missing)
                if can_add(mask, x):
                    mask |= (1 << (x - 1))
                    if bin(mask).count('1') > best_size:
                        best_size = bin(mask).count('1')
                        best = mask
            
            # Occasionally remove an element to allow others
            if random.random() < 0.15:
                present = [x for x in range(1, n + 1) if (mask >> (x - 1)) & 1]
                if present:
                    x = random.choice(present)
                    mask &= ~(1 << (x - 1))
        
        return best
    
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different random orderings
    remaining = deadline - time.time()
    num_restarts = max(3, int(remaining / 1.5))
    
    for r in range(num_restarts):
        if time.time() >= deadline - 2:
            break
        
        # Random ordering
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Greedy MIS
        mask = greedy_mis(order)
        sz = bin(mask).count('1')
        
        # Local search with remaining time budget
        ls_time = min(1.0, (deadline - time.time()) / max(1, num_restarts - r))
        if ls_time > 0.1:
            mask = local_search(mask, ls_time)
            sz = bin(mask).count('1')
        
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final greedy pass on best
    for x in range(1, n + 1):
        if not (best_mask >> (x - 1)) & 1:
            if can_add(best_mask, x):
                best_mask |= (1 << (x - 1))
    
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Final verification
    result_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                s1, s2, s3 = a + b, a + c, b + c
                if s1 in result_set and s2 in result_set and s3 in result_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(65, 129))
    
    return result
