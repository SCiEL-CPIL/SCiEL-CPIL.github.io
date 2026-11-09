import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
            
        valid = True
        for a in S:
            if (x + a) in sums:
                valid = False
                break
        
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def fast_reduce_maximal(S, N, deadline):
    """
    Optimized reduction of a maximal Sidon set.
    Uses blocker analysis to quickly identify non-critical elements for removal.
    """
    S = list(S)
    if not S:
        return S

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        S_set = set(S)
        
        # Precompute sums
        sums = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                sums.add(a + S[j])
        
        # Compute blockers for each y not in S
        # blockers[y] = list of a in S such that y+a is in sums
        # If a y has no blockers, S is not maximal (shouldn't happen if we start maximal)
        # If a y has exactly one blocker x, x is critical and cannot be removed.
        
        blockers = {}
        for y in range(1, N + 1):
            if y in S_set:
                continue
            bys = []
            for a in S:
                if (y + a) in sums:
                    bys.append(a)
            if not bys:
                # Not maximal? Should not happen in our flow, but handle gracefully
                # If this happens, we can't reduce further or S is invalid
                return S
            blockers[y] = bys
        
        # Identify critical elements
        critical = set()
        for y, bys in blockers.items():
            if len(bys) == 1:
                critical.add(bys[0])
        
        # Candidates are non-critical elements
        candidates = [x for x in S if x not in critical]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Try removing x
            candidate_S = [s for s in S if s != x]
            
            # Quick check: is candidate_S still maximal?
            # We know candidate_S is Sidon (subset of Sidon set).
            # We need to check maximality.
            # Optimization: We already know that for all y not in S, y is blocked by S.
            # When we remove x, y might become unblocked if x was the *only* blocker 
            # for y in S\{x} context? 
            # Actually, the blocker logic above was for S.
            # If x is not critical, it means for every y where x was a blocker, 
            # there is another blocker a != x.
            # BUT, the sum y+a might be formed by x in S\{x}? No, sums are from pairs in S\{x}.
            # The condition for y to be blocked in S\{x} is:
            # exists a in S\{x} such that y+a is in sums_{S\{x}}.
            # sums_{S\{x}} = {u+v | u,v in S\{x}, u<=v}.
            # We know y+a is in sums_S.
            # If y+a is formed by a pair in S\{x}, then it is in sums_{S\{x}}.
            # If y+a is formed only by pairs involving x in S, then it is NOT in sums_{S\{x}}.
            
            # So the "critical" check based on count of blockers in S is a necessary condition
            # but not sufficient for removability, because the blocker a might rely on a sum 
            # that involves x.
            
            # Therefore, we must verify is_maximal(candidate_S, N).
            # To speed this up, we can do a partial check.
            # But for now, let's just call is_maximal. It's O(N*|S|^2) ~ 256*400 = 100k ops.
            # With ~10 candidates, that's 1M ops. Very fast.
            
            if is_maximal(candidate_S, N):
                S = candidate_S
                improved = True
                break
                
    return S

def swap_search(S, N, deadline):
    """
    Try to improve S by swapping elements.
    """
    S = list(S)
    if not S:
        return S
    
    S_set = set(S)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try swapping each element with a random missing one
        for i, x in enumerate(S):
            if time.time() > deadline:
                break
            
            # Try a few random swaps
            for _ in range(10):
                y = random.randint(1, N)
                if y in S_set:
                    continue
                
                # Check if y can replace x
                candidate = S[:i] + S[i+1:] + [y]
                candidate.sort()
                
                if not is_sidon(candidate):
                    continue
                
                if is_maximal(candidate, N):
                    S = candidate
                    S_set = set(S)
                    improved = True
                    break
            
            if improved:
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = fast_reduce_maximal(S, N, deadline)
    if len(S) < best_size:
        best = S
        best_size = len(S)
        
    random.seed(42)
    
    # Try multiple random orderings
    # We can afford more trials now because reduction is faster
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce with remaining time budget
        remaining = deadline - time.time()
        if remaining > 5:
            S = fast_reduce_maximal(S, N, deadline)
        
        # Swap search
        if time.time() < deadline - 2:
            S = swap_search(S, N, deadline)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
            # If we found a very small set, we can tighten the search
            # e.g., if best_size <= 15, we might want to focus more on local search around best?
            # For now, just keep trying random restarts.
            
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
