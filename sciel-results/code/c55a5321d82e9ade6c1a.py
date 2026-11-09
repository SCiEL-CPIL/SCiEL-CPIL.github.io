import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin
    
    # Precompute conflict masks using bit-parallel representation
    # Each element is represented by a bit in a 128-bit integer
    # conflict_mask[u][v] = bitmask of all w that would form a forbidden triple with u and v
    
    # For three distinct elements a, b, c to be forbidden:
    # a+b, a+c, b+c must all be in the set
    # 
    # When we add x to set S, we need to check if there exist a, b in S such that:
    # Case 1: {a, b, x} is a forbidden triple
    #   - x+a in S, x+b in S, a+b in S (or a+b == x)
    # Case 2: x is one of the sums
    #   - x = a+b, a+c in S, b+c in S for some c in S
    #   - x = a+c, a+b in S, b+c in S for some b in S  
    #   - x = b+c, a+b in S, a+c in S for some a in S
    
    # Let's precompute for each pair (u, v), what third elements w would create a forbidden triple
    # A forbidden triple {a, b, c} means: a+b, a+c, b+c are all in the set
    # 
    # When adding x, conflicts arise when:
    # 1. x is a member: need a, b in S with x+a in S, x+b in S, a+b in S
    # 2. x is a sum: 
    #    - x = a+b: need c in S with a+c in S, b+c in S
    #    - x = a+c: need b in S with a+b in S, b+c in S
    #    - x = b+c: need a in S with a+b in S, a+c in S
    
    # For efficient checking, let's precompute:
    # For each x, define conflict_pairs[x] = set of pairs (a, b) such that adding x creates a forbidden triple
    # But this is complex. Let's use a different approach.
    
    # Actually, let's precompute for each element x, the set of "conflict witnesses"
    # When we add x to S, it's invalid if there exist a, b in S such that:
    # (x+a in S and x+b in S and a+b in S) OR
    # (a+b == x and a+c in S and b+c in S for some c in S) OR
    # (a+c == x and a+b in S and b+c in S for some b in S) OR
    # (b+c == x and a+b in S and a+c in S for some a in S)
    
    # Let's precompute a 128-bit mask for each element x that represents which other elements
    # would conflict with x given the current set S.
    
    # Actually, the cleanest approach: for each pair (u, v), precompute the set of w
    # such that {u, v, w} forms a forbidden triple.
    
    forbidden_triples = {}  # (u, v) -> set of w
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = a + b
            if ab > n:
                continue
            # Now we need c such that a+c, b+c are also in [1, n]
            # And {a, b, c} with a+b, a+c, b+c all in the set
            # c must be distinct from a and b
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac > n or bc > n:
                    continue
                # {a, b, c} is a forbidden triple if ab, ac, bc are all selected
                # Store this
                key = (a, b)
                if key not in forbidden_triples:
                    forbidden_triples[key] = set()
                forbidden_triples[key].add(c)
    
    # Now, for each element x, precompute which pairs (u, v) in the set would conflict with x
    # When adding x, it conflicts if:
    # 1. There exist u, v in S such that x, u, v form a forbidden triple
    # 2. x is a sum of two elements in S that have a common "witness"
    
    # Let's precompute for each x:
    # conflict_as_member[x] = set of pairs (u, v) such that {x, u, v} is a forbidden triple
    # conflict_as_sum[x] = set of pairs (u, v) such that u+v == x and there exists w with u+w, v+w in range
    
    conflict_as_member = {}
    conflict_as_sum = {}
    
    for x in range(1, n + 1):
        conflict_as_member[x] = set()
        conflict_as_sum[x] = set()
    
    # Fill conflict_as_member
    for (a, b), c_set in forbidden_triples.items():
        for c in c_set:
            # {a, b, c} is a forbidden triple
            conflict_as_member[a].add((b, c))
            conflict_as_member[b].add((a, c))
            conflict_as_member[c].add((a, b))
    
    # Fill conflict_as_sum
    # x = u + v, need w such that u+w in [1,n] and v+w in [1,n]
    for x in range(1, n + 1):
        for u in range(1, x):
            v = x - u
            if v <= u:  # Avoid duplicates, u < v
                continue
            if v > n:
                continue
            # Check if there exists w such that u+w <= n and v+w <= n
            # w can be anything in [1, n] except u and v, with u+w <= n and v+w <= n
            # This is always possible if u+w <= n for some w, which is true if u < n
            # So if u < n and v < n, there exists such w
            if u < n and v < n:
                conflict_as_sum[x].add((u, v))
    
    # Base set: {65, ..., 128}
    # Any a, b in this set have a+b >= 130 > 128, so a+b not in S.
    # Thus no triple with two elements from this set is forbidden.
    base_set = set(range(65, n + 1))
    candidates = list(range(1, 65))
    
    best_size = len(base_set)
    best_set = base_set.copy()
    
    # Bit-parallel representation
    # Each element i (1..128) corresponds to bit (i-1)
    def elem_to_bit(i):
        return 1 << (i - 1)
    
    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= elem_to_bit(x)
        return mask
    
    def mask_contains(mask, x):
        return (mask & elem_to_bit(x)) != 0
    
    def is_valid_addition_mask(S_mask, x):
        """
        Check if adding x to set represented by S_mask keeps it valid.
        x is not in S_mask.
        """
        # Case 1: x is a member of a forbidden triple {x, a, b}
        # Need a, b in S such that:
        # x+a in S, x+b in S, a+b in S
        # From conflict_as_member[x], we have pairs (a, b) such that {x, a, b} is forbidden
        # But we also need x+a in S and x+b in S
        # Actually, conflict_as_member[x] already encodes that {x, a, b} is a forbidden triple
        # which means x+a, x+b, a+b are all "potentially" in the set
        # We need to check if x+a, x+b, a+b are actually in S
        
        for (a, b) in conflict_as_member[x]:
            # Check if a, b, x+a, x+b, a+b are all in S (and x is being added)
            if mask_contains(S_mask, a) and mask_contains(S_mask, b):
                xa = x + a
                xb = x + b
                ab = a + b
                if xa <= n and xb <= n and ab <= n:
                    if mask_contains(S_mask, xa) and mask_contains(S_mask, xb) and mask_contains(S_mask, ab):
                        return False
        
        # Case 2: x is a sum
        # x = a + b, need c in S such that a+c in S and b+c in S
        for (a, b) in conflict_as_sum[x]:
            if mask_contains(S_mask, a) and mask_contains(S_mask, b):
                # Need to check if there exists c in S such that a+c in S and b+c in S
                # Iterate through possible c
                # c must be in S, c != a, c != b, a+c <= n, b+c <= n
                # To speed this up, we can precompute for each pair (a, b) the set of valid c's
                # But that's expensive. Let's just iterate.
                # Actually, we can be smarter: for each c in S, check if a+c and b+c are in S
                # This is O(|S|) per pair, which could be slow
                # Alternative: precompute for each (a, b) the bitmask of c such that a+c and b+c are both <= n
                # Then check if S_mask has any c in that mask, excluding a and b
                
                # For now, let's just iterate through S
                # But S is represented as a mask, so we need to extract elements
                # This is inefficient. Let's keep a set alongside the mask.
                pass
        
        return True
    
    # Let's use a hybrid approach: keep both a set and a mask for fast operations
    # Actually, let's just use sets but with precomputed conflict lists for faster checking
    
    def is_valid_addition(S, x):
        """
        Check if adding x to set S keeps S valid.
        S is assumed to be valid.
        x is not in S.
        """
        # Case 1: x is a member of the triple {x, a, b}
        # Need a, b in S such that:
        # x+a in S, x+b in S, a+b in S
        
        for (a, b) in conflict_as_member[x]:
            if a in S and b in S:
                xa = x + a
                xb = x + b
                ab = a + b
                if xa <= n and xb <= n and ab <= n:
                    if xa in S and xb in S and ab in S:
                        return False
        
        # Case 2: x is a sum
        # x = a + b, need c in S such that a+c in S and b+c in S
        for (a, b) in conflict_as_sum[x]:
            if a in S and b in S:
                # Need to find c in S such that a+c in S and b+c in S
                # Iterate through S
                for c in S:
                    if c == a or c == b:
                        continue
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ac in S and bc in S:
                        return False
        
        return True
    
    def try_add(S, x):
        if x in S:
            return S, False
        if is_valid_addition(S, x):
            S.add(x)
            return S, True
        return S, False
    
    # Strategy: Multiple random restarts with greedy addition
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Randomize candidate order
        c = candidates.copy()
        random.shuffle(c)
        
        current_set = base_set.copy()
        
        # Greedy add
        for x in c:
            if x in current_set:
                continue
            current_set, _ = try_add(current_set, x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        # If we have time, do a quick local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Try swapping
                for y in list(current_set):
                    # Remove y
                    current_set.remove(y)
                    
                    # Try to add any candidate
                    added_any = False
                    for z in c:
                        if z in current_set:
                            continue
                        new_set, added = try_add(current_set, z)
                        if added:
                            current_set = new_set
                            added_any = True
                            break
                            
                    if added_any:
                        improved = True
                    else:
                        # Put y back
                        current_set.add(y)
    
    return list(best_set)
