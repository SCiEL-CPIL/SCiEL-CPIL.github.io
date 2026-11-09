"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S):
    """Check if set S has no triple a<b<c with a+b, a+c, b+c all in S."""
    Sset = set(S)
    for i, a in enumerate(S):
        for b in S[i+1:]:
            ab = a + b
            if ab not in Sset:
                continue
            for c in S:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac in Sset and bc in Sset:
                    return False
    return True

def _can_add(Sset, x):
    """Check if adding x to Sset would create a forbidden triple."""
    # A forbidden triple involving x: either x is one of a,b,c or x is one of the sums
    # Case 1: x is one of a,b,c. WLOG x=c (largest). Then for some a<b in S:
    #   a+b in S, a+x in S, b+x in S
    # Case 2: x is one of the sums. WLOG x=a+b. Then for some a<b in S with a+b=x:
    #   for some c in S (c!=a,b): a+c in S and b+c in S
    # We check all cases.
    Slist = list(Sset)
    n = len(Slist)
    # Case 1: x plays role of c (or a or b). Check all pairs (a,b) in S:
    # If x is c: need a+b, a+x, b+x all in S (after adding x, a+x and b+x must be in S)
    # If x is a: need x+b, x+c, b+c all in S -> for some b,c in S: x+b in S, x+c in S, b+c in S
    # If x is b: need a+x, b+c=a+c... wait let me think more carefully.
    # Forbidden: three distinct members p,q,r such that p+q, p+r, q+r all in S.
    # After adding x, a new forbidden triple could involve x as a member or as a sum.
    # 
    # Sub-case A: x is one of the three members. WLOG x=r (the largest member).
    #   Then for some p<q in S: p+q in S, p+x in S, q+x in S.
    # Sub-case B: x is one of the three sums. WLOG x=p+q for some p<q in S.
    #   Then for some r in S (r distinct from p,q): p+r in S and q+r in S.
    #
    # Also x could be the middle member: x=q. Then for some p<x<r: p+q in S, p+r in S, q+r in S.
    # But since we're checking "would adding x create a violation", and x is new, we need:
    # - x as member: check all pairs (p,q) in S where x would be the third member
    # - x as sum: check all pairs (p,q) in S with p+q=x, then check if any r works
    
    # Efficient check:
    # A) x as third member (largest): for each pair (p,q) in S with p<q:
    #    if p+q in Sset and p+x in Sset and q+x in Sset -> invalid
    #    But p+x and q+x are > x, so they must already be in Sset (since x is being added, 
    #    p+x > x means p+x was already in S or not). Actually p+x > x, so p+x must be in Sset already.
    # B) x as sum: for each pair (p,q) in S with p+q == x:
    #    for each r in S, r != p, r != q: if p+r in Sset and q+r in Sset -> invalid
    # C) x as middle member: for each pair (p,r) in S with p < x < r:
    #    if p+x in Sset and p+r in Sset and x+r in Sset -> invalid
    
    # This is O(n^2) per candidate which is fine for n~70.
    for i in range(n):
        p = Slist[i]
        for j in range(i+1, n):
            q = Slist[j]
            # A) x is the largest member (r=x): need p+q, p+x, q+x all in S
            if p + q in Sset and p + x in Sset and q + x in Sset:
                return False
            # C) x is middle member: p < x < q or q < x < p
            #    WLOG p < x < q: need p+x, p+q, x+q all in S
            if p < x < q:
                if p + x in Sset and p + q in Sset and x + q in Sset:
                    return False
            elif q < x < p:
                if q + x in Sset and p + q in Sset and x + p in Sset:
                    return False
            # B) x is a sum: p+q == x
            if p + q == x:
                for r in Slist:
                    if r == p or r == q:
                        continue
                    if p + r in Sset and q + r in Sset:
                        return False
    return True

def _greedy_construct():
    """Greedy: try to add numbers from a random permutation."""
    Sset = set()
    candidates = list(range(1, 129))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(Sset, x):
            Sset.add(x)
    return Sset

def _local_search(Sset, time_limit=45):
    """Local search with add, remove, and swap moves."""
    best = Sset
    best_size = len(Sset)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        Slist = list(Sset)
        random.shuffle(Slist)
        
        # Try adding elements not in S
        not_in_S = [x for x in range(1, 129) if x not in Sset]
        random.shuffle(not_in_S)
        for x in not_in_S:
            if _can_add(Sset, x):
                Sset.add(x)
                improved = True
                if len(Sset) > best_size:
                    best = set(Sset)
                    best_size = len(Sset)
        
        # Try swap: remove one, add another
        Slist2 = list(Sset)
        random.shuffle(Slist2)
        for a in Slist2:
            if time.time() - start > time_limit - 5:
                break
            Sset.remove(a)
            not_in_S2 = [x for x in range(1, 129) if x not in Sset]
            random.shuffle(not_in_S2)
            added = False
            for x in not_in_S2:
                if _can_add(Sset, x):
                    Sset.add(x)
                    added = True
                    if len(Sset) > best_size:
                        best = set(Sset)
                        best_size = len(Sset)
                    break
            if not added:
                Sset.add(a)  # put it back
        
        if not improved:
            # Try some random restarts of local moves
            for _ in range(20):
                if time.time() - start > time_limit - 5:
                    break
                # Randomly remove a few and re-add
                to_remove = random.sample(list(Sset), min(3, len(Sset)))
                for a in to_remove:
                    Sset.remove(a)
                not_in_S3 = [x for x in range(1, 129) if x not in Sset]
                random.shuffle(not_in_S3)
                for x in not_in_S3:
                    if _can_add(Sset, x):
                        Sset.add(x)
                        if len(Sset) > best_size:
                            best = set(Sset)
                            best_size = len(Sset)
    
    return best

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    
    best = set()
    best_size = 0
    
    # Multiple greedy starts with different seeds
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        Sset = _greedy_construct()
        if len(Sset) > best_size:
            best = set(Sset)
            best_size = len(Sset)
    
    # Local search on best
    if time.time() < deadline - 5:
        improved = _local_search(set(best), time_limit=min(45, deadline - time.time() - 5))
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # A few more local search rounds
    while time.time() < deadline - 5:
        improved = _local_search(set(best), time_limit=min(15, deadline - time.time() - 5))
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
        else:
            break
    
    return sorted(best)
# EVOLVE-BLOCK-END
