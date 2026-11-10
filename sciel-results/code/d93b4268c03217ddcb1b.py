import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 88.0
    n = 128
    U = set(range(65, 129))
    
    # Precompute: for each element x in 1..64, and each pair (b,c) in 1..64,
    # determine whether adding x to L would create a forbidden triple.
    # A forbidden triple (a,b,c) with a,b,c in L requires:
    #   a+b in S, a+c in S, b+c in S
    # where S = L ∪ U.
    # s in S iff s in L or s >= 65.
    
    # For incremental checking: when we add element x to L, we need to check
    # all triples (x, b, c) where b, c are already in L.
    # Also triples (a, b, c) already in L are unaffected (their sums don't change).
    # So we only need to check triples involving the new element x.
    
    # Precompute for each x, the set of (b, c) pairs such that if x, b, c are all in L,
    # then the triple is forbidden given the current L.
    # But this depends on current L, so we can't fully precompute.
    
    # Instead, for each x, precompute all pairs (b, c) with b < c, b != x, c != x,
    # such that:
    #   - a+b, a+c, b+c would all be in S if x, b, c are in L
    #   where a is the smallest of (x, b, c)
    
    # Simpler approach: for each x, and each pair (b, c) in 1..64, b < c,
    # determine if (x, b, c) as a triple would be forbidden when L contains {x, b, c}
    # and U is always present.
    
    # Triple (a, b, c) with a < b < c is forbidden if:
    #   (a+b in L or a+b >= 65) AND (a+c in L or a+c >= 65) AND (b+c in L or b+c >= 65)
    
    # For incremental check when adding x:
    # For each pair (b, c) in current L, check if (min(x,b,c), mid, max) is forbidden.
    
    def triple_forbidden(a, b, c, L_set):
        """Check if triple (a,b,c) is forbidden given L_set and U always present."""
        s1, s2, s3 = a+b, a+c, b+c
        def in_S(s):
            return s in L_set or s >= 65
        return in_S(s1) and in_S(s2) and in_S(s3)
    
    def can_add(x, L_set):
        """Check if adding x to L_set keeps it valid. Only check triples involving x."""
        for b in L_set:
            if b == x:
                continue
            for c in L_set:
                if c == x or c <= b:
                    continue
                # Triple is (x, b, c) in some order
                a, d, e = sorted([x, b, c])
                if triple_forbidden(a, d, e, L_set):
                    return False
        return True
    
    def is_valid_L(L):
        L_set = set(L)
        for a, b, c in combinations(L, 3):
            if triple_forbidden(a, b, c, L_set):
                return False
        return True
    
    # Start with base solution
    L_base = list(range(17, 33))
    L_set = set(L_base)
    
    # Verify base is valid
    assert is_valid_L(L_base), "Base solution should be valid"
    
    random.seed(42)
    candidates = [x for x in range(1, 65) if x not in L_set]
    
    # Phase 1: Greedy addition
    changed = True
    while changed and time.time() < deadline:
        changed = False
        random.shuffle(candidates)
        for x in candidates[:]:
            if x in L_set:
                continue
            if can_add(x, L_set):
                L_set.add(x)
                L_base.append(x)
                candidates.remove(x)
                changed = True
    
    best_L = list(L_set)
    best_size = len(best_L)
    
    # Phase 2: Local search with swaps and 2-for-1 exchanges
    remaining = [x for x in range(1, 65) if x not in L_set]
    
    for iteration in range(5000):
        if time.time() > deadline - 2.0:
            break
        if not remaining:
            break
        
        # Try single addition
        if remaining:
            x = random.choice(remaining)
            if can_add(x, L_set):
                L_set.add(x)
                best_L = list(L_set)
                best_size = len(best_L)
                remaining.remove(x)
                continue
        
        # Try single swap: remove one from L, add one from remaining
        if remaining and L_set:
            to_remove = random.choice(list(L_set))
            to_add = random.choice(remaining)
            new_L_set = L_set - {to_remove} | {to_add}
            if is_valid_L(list(new_L_set)):
                L_set = new_L_set
                best_L = list(L_set)
                remaining.remove(to_add)
                remaining.append(to_remove)
                continue
        
        # Try 2-for-1: remove two from L, add one from remaining (net -1, but might enable future gains)
        # Actually, let's try 1-for-2: remove one, add two
        if len(remaining) >= 2 and L_set:
            to_remove = random.choice(list(L_set))
            r1, r2 = random.sample(remaining, 2)
            new_L_set = L_set - {to_remove} | {r1, r2}
            if is_valid_L(list(new_L_set)):
                L_set = new_L_set
                best_L = list(L_set)
                remaining.remove(r1)
                remaining.remove(r2)
                remaining.append(to_remove)
                continue
    
    # Phase 3: Try to add any remaining elements
    for x in remaining[:]:
        if time.time() > deadline - 1.0:
            break
        if can_add(x, L_set):
            L_set.add(x)
            remaining.remove(x)
    
    best_L = list(L_set)
    S = sorted(list(U | L_set))
    
    # Final verification
    S_set = set(S)
    valid = True
    for a, b, c in combinations(S, 3):
        if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
            valid = False
            break
    
    if not valid:
        # Fallback to base solution
        return [int(x) for x in sorted(list(range(65, 129)) + list(range(17, 33)))]
    
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
