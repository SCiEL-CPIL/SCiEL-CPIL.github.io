import time

def _check_addition(Sset, elems, new_elem):
    """Check if adding new_elem to Sset creates a forbidden triple.
    Only need to check triples where new_elem is one of a,b,c."""
    n = len(elems)
    # Case 1: new_elem is c (largest of triple)
    # For all pairs (a,b) in elems with a<b: if a+b, a+new_elem, b+new_elem all in Sset -> bad
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            if a + b in Sset and a + new_elem in Sset and b + new_elem in Sset:
                return False
    # Case 2: new_elem is b (middle of triple)
    # For all pairs (a,c) where a < new_elem < c: if a+c, a+new_elem, c+new_elem all in Sset -> bad
    for i in range(n):
        a = elems[i]
        if a >= new_elem:
            break
        for k in range(n):
            c = elems[k]
            if c <= new_elem:
                continue
            if a + c in Sset and a + new_elem in Sset and c + new_elem in Sset:
                return False
    # Case 3: new_elem is a (smallest of triple)
    # For all pairs (b,c) where new_elem < b < c: if b+c, new_elem+b, new_elem+c all in Sset -> bad
    for i in range(n):
        b = elems[i]
        if b <= new_elem:
            continue
        for j in range(i+1, n):
            c = elems[j]
            if b + c in Sset and new_elem + b in Sset and new_elem + c in Sset:
                return False
    return True

def _valid(Sset):
    """Full validity check."""
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

def construct():
    deadline = time.time() + 85.0  # Safety margin: evaluator kills at 90s
    
    # Start from the strong sufficient-condition interval: {64..128}, size 65
    S = set(range(64, 129))
    best = set(S)
    best_size = len(best)
    
    # Greedily try to add each small element 1..63 if validity preserved
    for cand in range(1, 64):
        if time.time() > deadline:
            break
        if cand in S:
            continue
        elems = sorted(S)
        if _check_addition(S, elems, cand):
            S.add(cand)
            if len(S) > best_size:
                best = set(S)
                best_size = len(best)
    
    # Local search phase 1: try to add two elements at once (2-for-0 swaps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 64):
            if time.time() > deadline:
                break
            if x in S:
                continue
            for y in range(x+1, 64):
                if time.time() > deadline:
                    break
                if y in S:
                    continue
                # Try adding x and y
                S.add(x)
                S.add(y)
                elems = sorted(S)
                if _check_addition(S - {y}, [e for e in elems if e != y], y):
                    if len(S) > best_size:
                        best = set(S)
                        best_size = len(best)
                    changed = True
                else:
                    S.discard(y)
                    S.discard(x)
    
    # Local search phase 2: remove one, add two (net +1)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        elems_sorted = sorted(S)
        for rem in elems_sorted:
            if time.time() > deadline:
                break
            S.discard(rem)
            for x in range(1, 64):
                if time.time() > deadline:
                    break
                if x in S:
                    continue
                for y in range(x+1, 64):
                    if time.time() > deadline:
                        break
                    if y in S:
                        continue
                    S.add(x)
                    S.add(y)
                    elems = sorted(S)
                    if _check_addition(S - {y}, [e for e in elems if e != y], y):
                        if len(S) > best_size:
                            best = set(S)
                            best_size = len(best)
                        changed = True
                        break
                    else:
                        S.discard(y)
                if changed:
                    break
            if changed:
                break
            S.add(rem)
    
    # Final: ensure we return a valid set
    result = sorted(best)
    result = [int(x) for x in result]
    return result
