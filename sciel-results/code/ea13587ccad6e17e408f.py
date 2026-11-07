import time
import random

def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Bitmask helpers
    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]
    
    def list_to_mask(lst):
        m = 0
        for v in lst:
            m |= (1 << (v - 1))
        return m
    
    def has(mask, v):
        return (mask >> (v - 1)) & 1
    
    def is_valid(mask):
        elems = mask_to_list(mask)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    break
                if not has(mask, ab):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        break
                    if has(mask, ac) and has(mask, bc):
                        return False
        return True
    
    # Start with upper half: 65..128 (all pair sums > 128, so internally safe)
    base_mask = 0
    for v in range(65, 129):
        base_mask |= (1 << (v - 1))
    
    best_mask = base_mask
    best_size = 64
    
    # Greedy: try adding lower elements 1..64
    # For each candidate v in 1..64, check if adding it creates a forbidden triple
    def can_add(mask, v):
        """Check if v can be added to mask without creating a forbidden triple."""
        if has(mask, v):
            return True
        # A forbidden triple (a,b,c) with a<b<c requires a+b, a+c, b+c all in S
        # If we add v, v could be one of a,b,c or one of the sums
        # Case 1: v is a sum (a+b=v, a+c=v, b+c=v) - impossible since sums are distinct for distinct pairs
        # Case 2: v is one of the three elements a, b, or c
        # If v=a: need b,c in S with b+c in S and v+b in S and v+c in S
        #   i.e., exists b,c in S, b<c, b+c in S, v+b in S, v+c in S
        # If v=b: need a,c in S, a<c, a+c in S, a+v in S, v+c in S  
        # If v=c: need a,b in S, a<b, a+b in S, a+v in S, b+v in S
        
        elems = mask_to_list(mask)
        L = len(elems)
        
        for i in range(L):
            for j in range(i + 1, L):
                a = elems[i]
                b = elems[j]
                ab = a + b
                if ab > n:
                    break
                if not has(mask, ab):
                    continue
                # Now check if adding v creates a triple with (a,b)
                # v could be c: need a+v in S, b+v in S, and v > b
                if v > b:
                    av = a + v
                    bv = b + v
                    if av <= n and bv <= n and has(mask, av) and has(mask, bv):
                        return False
                # v could be a: need v < a, and v+b in S, v+c... wait, we need three distinct
                # Actually let's be more careful. The triple is (x,y,z) distinct with all pair sums in S.
                # If v is added, the new forbidden triples must include v as one of the three elements.
                # So we need to check: are there x,y in S (x≠y, x≠v, y≠v) such that
                #   {x,y,v} are distinct and x+y, x+v, y+v are all in S∪{v}
                # Since v is new, x+v and y+v must be in S (not equal to v since they're sums > v typically)
                # and x+y must be in S.
                
                # Let me redo: check all pairs (x,y) in S, x<y, and see if {x,y,v} forms forbidden triple
                # with all three pair sums in S∪{v}
        
        # More efficient: for each pair (a,b) in S with a+b in S, check if v completes a triple
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    break
                if not has(mask, ab):
                    continue
                # Check if (a,b,v) is a forbidden triple with all sums in S∪{v}
                # Sums: a+b (in S), a+v, b+v
                # Need a+v in S∪{v} and b+v in S∪{v}
                # a+v = v means a=0, impossible. So a+v must be in S and ≠ v
                # Similarly b+v must be in S
                av = a + v
                bv = b + v
                if av <= n and bv <= n:
                    if has(mask, av) and has(mask, bv):
                        # Check distinctness: a, b, v must be distinct
                        if a != b and a != v and b != v:
                            return False
        return True
    
    # Greedy add lower elements
    mask = base_mask
    for v in range(1, 65):
        if time.time() > deadline - 5:
            break
        if can_add(mask, v):
            mask |= (1 << (v - 1))
    
    if mask.bit_count() > best_size:
        best_size = mask.bit_count()
        best_mask = mask
    
    # Random order greedy
    rng = random.Random(42)
    for trial in range(300):
        if time.time() > deadline - 15:
            break
        m = base_mask
        remaining = list(range(1, 65))
        rng.shuffle(remaining)
        for v in remaining:
            if time.time() > deadline - 15:
                break
            if can_add(m, v):
                m |= (1 << (v - 1))
        sz = m.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Local search: try swapping elements
    if time.time() < deadline - 20:
        m = best_mask
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            elems = mask_to_list(m)
            out_elems = [v for v in range(1, n + 1) if not has(m, v)]
            for a in elems:
                if time.time() > deadline - 20:
                    break
                m2 = m & ~(1 << (a - 1))
                added = 0
                for b in out_elems:
                    if b == a:
                        continue
                    if can_add(m2, b):
                        m2 |= (1 << (b - 1))
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    if m2.bit_count() > best_size:
                        best_size = m2.bit_count()
                        best_mask = m2
                    improved = True
                    m = m2
                # else keep m as is
    
    # Final validation
    if is_valid(best_mask):
        result = mask_to_list(best_mask)
    else:
        # Fallback to safe upper half
        result = list(range(65, 129))
    
    return [int(v) for v in result]
