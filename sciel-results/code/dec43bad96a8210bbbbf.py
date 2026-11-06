"""Find maximum subset of [1,180] with no three distinct a,b,c where 1/a=1/b+1/c."""
import time
import itertools

def _find_forbidden_triples(N=180):
    """Find all triples (a,b,c) of distinct elements in [1,N] where 1/a=1/b+1/c."""
    triples = []
    for a in range(1, N+1):
        for b in range(1, N+1):
            if a == b:
                continue
            # c = a*b/(a+b), must be integer and distinct from a,b
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    triple = tuple(sorted((a, b, c)))
                    triples.append(triple)
    # Deduplicate
    return list(set(triples))

def _is_valid(S, forbidden_triples):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for t in forbidden_triples:
        if all(x in S_set for x in t):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 180
    forbidden = _find_forbidden_triples(N)
    
    # Build conflict structure: for each element, which triples it's in
    elem_triples = {i: [] for i in range(1, N+1)}
    for idx, t in enumerate(forbidden):
        for x in t:
            elem_triples[x].append(idx)
    
    # Greedy removal: start with all, remove elements to break all triples
    S = set(range(1, N+1))
    active_triples = set(range(len(forbidden)))
    
    while active_triples and time.time() < deadline:
        # Find element in most active triples
        counts = {}
        for ti in active_triples:
            for x in forbidden[ti]:
                if x in S:
                    counts[x] = counts.get(x, 0) + 1
        if not counts:
            break
        # Remove element with max count
        worst = max(counts, key=lambda x: counts[x])
        S.remove(worst)
        # Update active triples
        new_active = set()
        for ti in active_triples:
            if all(x in S for x in forbidden[ti]):
                new_active.add(ti)
        active_triples = new_active
    
    best = len(S)
    best_S = set(S)
    
    # Local search: try to add elements back
    removed = set(range(1, N+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(removed):
            if time.time() > deadline:
                break
            # Check if adding x creates any forbidden triple
            S_test = S | {x}
            valid = True
            for ti in range(len(forbidden)):
                t = forbidden[ti]
                if x in t and all(y in S_test for y in t):
                    valid = False
                    break
            if valid:
                S.add(x)
                removed.discard(x)
                improved = True
                if len(S) > best:
                    best = len(S)
                    best_S = set(S)
    
    # Try pairwise swaps
    if time.time() < deadline - 5:
        for _ in range(50):  # limited iterations
            if time.time() > deadline:
                break
            # Try swapping: remove one from S, add one from removed
            found = False
            for x in list(S):
                if found or time.time() > deadline:
                    break
                for y in list(removed):
                    if time.time() > deadline:
                        break
                    S_test = (S - {x}) | {y}
                    valid = True
                    for ti in range(len(forbidden)):
                        t = forbidden[ti]
                        if all(z in S_test for z in t):
                            valid = False
                            break
                    if valid and len(S_test) > len(S):
                        S = S_test
                        removed = set(range(1, N+1)) - S
                        if len(S) > best:
                            best = len(S)
                            best_S = set(S)
                        found = True
                        break
    
    return sorted(best_S) if best_S else sorted(S)
