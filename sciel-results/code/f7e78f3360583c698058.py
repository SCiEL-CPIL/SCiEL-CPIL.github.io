import time

def _valid(Sset):
    # Sset: set of ints. Check no distinct a<b<c in Sset with a+b,a+c,b+c all in Sset.
    elems = sorted(Sset)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = elems[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def _check_addition(Sset, elems, new_elem):
    """Check if adding new_elem to Sset creates a forbidden triple.
    Only need to check triples where new_elem is one of a,b,c.
    elems must be sorted list of Sset elements."""
    n = len(elems)
    # Case 1: new_elem is c (largest in triple)
    # Check all pairs (a,b) in elems: if a+b, a+new_elem, b+new_elem all in Sset
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            if a + b in Sset and a + new_elem in Sset and b + new_elem in Sset:
                return False
    # Case 2: new_elem is b (middle in triple)
    # Check pairs (a, c) where a < new_elem < c
    for i in range(n):
        a = elems[i]
        if a >= new_elem:
            break
        for k in range(n):
            c = elems[k]
            if c <= new_elem:
                continue
            if a + new_elem in Sset and a + c in Sset and new_elem + c in Sset:
                return False
    # Case 3: new_elem is a (smallest in triple)
    # Check pairs (b, c) where new_elem < b < c
    for i in range(n):
        b = elems[i]
        if b <= new_elem:
            continue
        for j in range(i+1, n):
            c = elems[j]
            if new_elem + b in Sset and new_elem + c in Sset and b + c in Sset:
                return False
    return True

def construct():
    deadline = time.time() + 85.0
    # Start from the strong sufficient-condition interval: {64..128}, size 65
    S = set(range(64, 129))
    best = set(S)
    best_elems = sorted(S)
    
    # Greedily try to add each small element 1..63 if validity preserved
    for cand in range(1, 64):
        if cand in S:
            continue
        if _check_addition(S, best_elems, cand):
            S = S | {cand}
            best_elems = sorted(S)
            best = set(S)
    
    # Local search: try single swaps to increase size
    # Actually adding increases size by 1; we already maximized additions. 
    # Try 2-for-1: add two small, keep size non-decreasing while valid.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        base = time.time()
        for x in range(1, 64):
            if x in S:
                continue
            if _check_addition(S, best_elems, x):
                S = S | {x}
                best_elems = sorted(S)
                best = set(S)
                changed = True
        if time.time() - base > 50:
            break
    
    # Try adding pairs of elements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        base = time.time()
        for x in range(1, 64):
            if x in S:
                continue
            for y in range(x+1, 64):
                if y in S:
                    continue
                trial = S | {x, y}
                if _valid(trial):
                    S = trial
                    best_elems = sorted(S)
                    best = set(S)
                    changed = True
        if time.time() - base > 30:
            break
    
    # Ensure within deadline
    if time.time() > deadline:
        best = set(range(64, 129))
    
    result = sorted(best)
    # Type check: plain ints
    result = [int(x) for x in result]
    return result
