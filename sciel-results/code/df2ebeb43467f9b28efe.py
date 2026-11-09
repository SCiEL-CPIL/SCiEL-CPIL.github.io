import time, math, random

def _triples(n):
    # all triples (x,y,z) with 1<=x<y<z<=n and (y-x)(z-x)==x*x
    T=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d; z=x+e
                    if y<=n and z<=n and y!=z:
                        T.append((x,y,z))
            d+=1
    return T

def construct():
    t0=time.time()
    DEADLINE=85.0 # Safety margin for 90s limit
    n=180
    T=_triples(n)
    
    # Precompute for each element x, the list of triples containing x
    # triples_by_elem[x] = list of (other1, other2, triple_index)
    triples_by_elem = [[] for _ in range(n+1)]
    for idx, (a,b,c) in enumerate(T):
        triples_by_elem[a].append((b, c, idx))
        triples_by_elem[b].append((a, c, idx))
        triples_by_elem[c].append((a, b, idx))

    def is_conflict(S, x):
        """Check if adding x to S creates a conflict (i.e., completes a triple)."""
        for o1, o2, idx in triples_by_elem[x]:
            if S[o1] and S[o2]:
                return True
        return False

    def is_feasible(S):
        """Check if S contains any conflicting triple."""
        # We can check by iterating through all triples, but that's slow.
        # Better: check if any element in S has a conflict? No, that's not right.
        # If S is feasible, then for every x in S, is_conflict(S, x) is False?
        # No, is_conflict(S, x) checks if x *completes* a triple with two others in S.
        # If S is feasible, no triple is fully contained.
        # So for any x in S, there should be no pair (o1, o2) in S such that (x, o1, o2) is a triple.
        # This is equivalent to checking is_conflict(S, x) for all x in S?
        # Actually, if a triple (a,b,c) is in S, then for x=a, is_conflict(S, a) will see b and c in S and return True.
        # So yes, S is feasible iff for all x in S, is_conflict(S, x) is False.
        # But checking all x in S is O(|S| * deg(x)).
        # Alternatively, just iterate through T and check if all 3 are in S.
        # Since |T| is small (approx 1000-2000?), iterating T is fast enough.
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                return False
        return True

    def greedy_build(order):
        S = [False]*(n+1)
        for x in order:
            if not is_conflict(S, x):
                S[x] = True
        return S

    def local_search(S, time_limit):
        """Try to improve S by adding/removing elements."""
        S = S[:]
        current_size = sum(S)
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            # Try adding random elements not in S
            candidates = [i for i in range(1, n+1) if not S[i]]
            random.shuffle(candidates)
            for add in candidates:
                if is_conflict(S, add):
                    continue
                S[add] = True
                current_size += 1
                improved = True
                break
            if time.time() - t0 > time_limit:
                break
            # Try swapping: remove one, add another
            # This is expensive, so we do it less frequently or with a smaller scope
            # For now, let's stick to additions. Removals are rarely beneficial unless we want to enable a larger addition.
            # A better heuristic: if we can't add, try to remove an element that is part of many conflicts with non-S elements?
            # Or simply: try to remove a random element and see if we can add a new one.
            # Let's try a few random swaps.
            for _ in range(50):
                if time.time() - t0 > time_limit:
                    break
                # Pick a random element to remove
                in_set = [i for i in range(1, n+1) if S[i]]
                if not in_set:
                    break
                rem = random.choice(in_set)
                S[rem] = False
                # Try to add a random element not in S
                not_in_set = [i for i in range(1, n+1) if not S[i]]
                random.shuffle(not_in_set)
                added = False
                for add in not_in_set:
                    if not is_conflict(S, add):
                        S[add] = True
                        added = True
                        if sum(S) > current_size:
                            current_size = sum(S)
                            improved = True
                        break
                if not added:
                    # Restore removed element if we didn't improve size
                    S[rem] = True
        return S

    best_S = None
    best_size = 0
    
    # Strategy 1: Descending Greedy
    S1 = greedy_build(range(n, 0, -1))
    S1 = local_search(S1, DEADLINE - 10)
    if sum(S1) > best_size:
        best_size = sum(S1)
        best_S = S1
    
    # Strategy 2: Ascending Greedy
    if time.time() - t0 < DEADLINE - 10:
        S2 = greedy_build(range(1, n+1))
        S2 = local_search(S2, DEADLINE - 10)
        if sum(S2) > best_size:
            best_size = sum(S2)
            best_S = S2

    # Strategy 3: Random Orders
    while time.time() - t0 < DEADLINE - 5:
        random_order = list(range(1, n+1))
        random.shuffle(random_order)
        S3 = greedy_build(random_order)
        S3 = local_search(S3, DEADLINE - 5)
        if sum(S3) > best_size:
            best_size = sum(S3)
            best_S = S3

    if best_S is None:
        best_S = [False]*(n+1)
        best_S[180] = True

    out = [i for i in range(1, n+1) if best_S[i]]
    return out
