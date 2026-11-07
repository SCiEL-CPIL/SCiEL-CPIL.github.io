import math
import time

def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    """
    deadline = time.time() + 50
    
    # Phase 1: Start with a large antichain [129, 256]
    S = list(range(129, 257))
    
    def get_conflicts(S):
        """
        Returns a dictionary of conflict counts for each element in S.
        Also returns the list of violating triples if needed for debugging, 
        but we just need counts to pick the worst element.
        """
        counts = {x: 0 for x in S}
        S_set = set(S)
        
        # Optimization: Precompute gcd? No, math.gcd is C-implemented and fast.
        # Loop through all pairs (a, b)
        # For each pair, check if there exists c such that a | b*c
        # a | b*c  <=>  (a/gcd(a,b)) | c
        # Let req = a // gcd(a, b).
        # We need c in S, c != a, c != b, such that c % req == 0.
        
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i == j:
                    continue
                g = math.gcd(a, b)
                req = a // g
                if req == 1:
                    # a divides b. In an antichain this shouldn't happen, 
                    # but if S is not an antichain, it's a violation for any c.
                    # However, if a|b, then a | b*1. But c must be distinct.
                    # If a|b, then for ANY c distinct from a,b, a | b*c.
                    # So this is a massive source of conflicts.
                    # Count conflicts for a and b.
                    # How many c? |S| - 2.
                    # But we just need to know *if* there is a conflict to remove something.
                    # Actually, if a|b, then (a,b,c) is a violation for all c.
                    # So a and b are very bad.
                    # Let's just mark them as having conflicts.
                    if len(S) > 2:
                         counts[a] += 1
                         counts[b] += 1
                    continue
                
                # Find if any c in S satisfies c % req == 0
                # and c != a and c != b
                found = False
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % req == 0:
                        found = True
                        # We found a violation. 
                        # We can break early for this (a,b) pair?
                        # Yes, one violation is enough to increment counts.
                        # But we want to count *how many* to prioritize.
                        # Counting all is O(N) per pair. O(N^3) total.
                        # 128^3 is 2M. This is fine.
                        break
                
                if found:
                    counts[a] += 1
                    counts[b] += 1
                    # Note: We didn't increment c. 
                    # Should we? The violation is symmetric in the sense that 
                    # removing any of a,b,c fixes it.
                    # But a is the "divisor". 
                    # Usually, the divisor is the "problem" in divisibility graphs.
                    # However, if c is a multiple of req, c is "responsible" too.
                    # Let's increment c as well? 
                    # The triple (a,b,c) is bad.
                    # If we remove a, it's gone.
                    # If we remove b, it's gone.
                    # If we remove c, it's gone.
                    # So all three are involved.
                    # But we don't know which c specifically caused it if we broke early.
                    # Let's just count a and b. The "divisor" is the primary constraint violator.
                    # Actually, let's be more precise.
                    # If we count a and b, we might remove the wrong one if c is the "hub" of multiples.
                    # Let's try counting all three if we can identify them.
                    # For speed, let's stick to a and b for now. 
                    # Or, better: just count the pair (a,b) as a conflict.
                    # The element with the most "bad pairs" is the one to remove.
                    pass
        
        return counts

    # Greedy Removal
    while time.time() < deadline:
        if not S:
            break
            
        # Check if valid
        # We can integrate the validity check into the conflict counting.
        # If max count is 0, it's valid.
        
        conflicts = get_conflicts(S)
        max_conf = max(conflicts.values()) if conflicts else 0
        
        if max_conf == 0:
            break
        
        # Remove the element with the highest conflict count
        # Tie-breaking: remove the smallest? largest?
        # Let's remove the one with highest conflict, then smallest value (arbitrary)
        worst_elem = None
        max_val = -1
        for x in S:
            if conflicts[x] > max_val:
                max_val = conflicts[x]
                worst_elem = x
        
        if worst_elem is not None:
            S.remove(worst_elem)
        else:
            break

    # Phase 2: Greedy Addition
    # Try to add elements from 128 down to 2
    # Order: Larger numbers are "safer" (less likely to divide others' products)
    # and "less likely to be divided by" others?
    # Actually, if x is large, x | b*c is harder to satisfy (needs large b,c).
    # And a | x*c is easier to satisfy?
    # If a is small, a | x*c is likely.
    # So adding large x is safer.
    
    for x in range(128, 1, -1):
        if time.time() > deadline:
            break
            
        if x in S:
            continue
            
        # Check if S U {x} is valid
        # Only need to check triples involving x.
        valid = True
        S_set = set(S)
        
        # Triples: (x, b, c), (b, x, c), (b, c, x) where b,c in S, b!=c
        # 1. x | b*c
        # 2. b | x*c
        # 3. c | x*b  (same as 2 with b,c swapped)
        
        # Iterate b, c in S
        # To optimize, we can break early.
        
        for i, b in enumerate(S):
            for j, c in enumerate(S):
                if i == j:
                    continue
                
                # Check 1: x | b*c
                if (b * c) % x == 0:
                    valid = False
                    break
                
                # Check 2: b | x*c
                if (x * c) % b == 0:
                    valid = False
                    break
                
                # Check 3: c | x*b
                if (x * b) % c == 0:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            S.append(x)

    return S
