"""Strategy: Multi-restart greedy + local search for maximum set avoiding reciprocal Schur triples."""
import time
import random

def _build_conflicts(N):
    """Build for each number x, the list of (y, z) pairs such that {x, y, z} is a forbidden triple."""
    # For each a in [1, N], find all pairs (b, c) with b, c distinct, b != a, c != a,
    # such that 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2
    # Let d1 = b - a, d2 = c - a, then d1 * d2 = a^2
    # We need b = a + d1 in [1, N], c = a + d2 in [1, N], b != c, b != a, c != a
    # Also d1 and d2 can be negative (if b < a or c < a)
    
    # For each a, enumerate all factor pairs (d1, d2) of a^2 where d1*d2 = a^2
    # d1 and d2 can be positive or negative
    # b = a + d1, c = a + d2
    # Need 1 <= b <= N, 1 <= c <= N, b != a, c != a, b != c
    
    # conflict[x] = list of (y, z) such that {x, y, z} is a triple
    # We'll build it as: for each triple (a, b, c), add to conflict[a], conflict[b], conflict[c]
    
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d of a^2 (positive)
        # For each positive divisor d, we get pairs (d, a2//d) and (-d, -a2//d)
        # Also need to consider that d1 can be any divisor (positive or negative)
        
        # Enumerate all divisors of a^2
        divisors = []
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        
        for d1 in divisors:
            d2 = a2 // d1
            # Case 1: both positive
            b1 = a + d1
            c1 = a + d2
            if 1 <= b1 <= N and 1 <= c1 <= N and b1 != a and c1 != a and b1 != c1:
                triple = tuple(sorted((a, b1, c1)))
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)
            # Case 2: both negative
            b2 = a - d1
            c2 = a - d2
            if 1 <= b2 <= N and 1 <= c2 <= N and b2 != a and c2 != a and b2 != c1:
                triple = tuple(sorted((a, b2, c2)))
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)
            # Case 3: d1 positive, d2 negative (but d1*d2 = a^2 > 0, so both same sign)
            # So only same-sign cases exist
        
        # Also need to handle the case where d1 = d2 (i.e., d1^2 = a^2, so d1 = a)
        # This is already covered since a is a divisor of a^2
    
    # Build conflict lookup: for each pair (x, y), the set of z such that {x, y, z} is a triple
    pair_to_z = {}
    for (a, b, c) in triples:
        for x, y, z in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
            key = (min(x, y), max(x, y))
            if key not in pair_to_z:
                pair_to_z[key] = set()
            pair_to_z[key].add(z)
    
    return triples, pair_to_z

def _can_add(x, S, pair_to_z):
    """Check if x can be added to S without creating a forbidden triple."""
    for y in S:
        if y == x:
            continue
        key = (min(x, y), max(x, y))
        if key in pair_to_z:
            for z in pair_to_z[key]:
                if z in S and z != x and z != y:
                    return False
    return True

def _greedy_independent_set(N, pair_to_z, order, deadline):
    """Greedy: add numbers in given order if they don't create a triple."""
    S = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, S, pair_to_z):
            S.add(x)
    return S

def _local_search(S, N, pair_to_z, deadline):
    """Local search: try swaps to improve the set size."""
    improved = True
    best_S = set(S)
    best_size = len(best_S)
    
    while improved and time.time() < deadline:
        improved = False
        current_size = len(best_S)
        
        # Try removing one element and adding another
        for x in sorted(best_S):
            if time.time() > deadline:
                break
            S_test = best_S - {x}
            
            # Try adding each y not in S_test
            for y in range(1, N + 1):
                if y in S_test:
                    continue
                if time.time() > deadline:
                    break
                if _can_add(y, S_test, pair_to_z):
                    S_new = S_test | {y}
                    if len(S_new) > best_size:
                        best_S = S_new
                        best_size = len(S_new)
                        improved = True
                        break
        
        # Try 2-swap: remove two elements, add two different ones
        if not improved and time.time() < deadline:
            elements = sorted(best_S)
            for i in range(len(elements)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(elements)):
                    if time.time() > deadline:
                        break
                    x1, x2 = elements[i], elements[j]
                    S_test = best_S - {x1, x2}
                    
                    # Try adding two new elements
                    for y1 in range(1, N + 1):
                        if y1 in S_test:
                            continue
                        if not _can_add(y1, S_test, pair_to_z):
                            continue
                        S_mid = S_test | {y1}
                        for y2 in range(y1 + 1, N + 1):
                            if y2 in S_mid:
                                continue
                            if time.time() > deadline:
                                break
                            if _can_add(y2, S_mid, pair_to_z):
                                S_new = S_mid | {y2}
                                if len(S_new) > best_size:
                                    best_S = S_new
                                    best_size = len(S_new)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 180
    random.seed(42)
    
    triples, pair_to_z = _build_conflicts(N)
    
    # Multi-restart greedy with different orderings
    best_S = set()
    best_size = 0
    
    # Restart 1: descending order
    S1 = _greedy_independent_set(N, pair_to_z, list(range(N, 0, -1)), deadline)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Restart 2: ascending order
    if time.time() < deadline - 10:
        S2 = _greedy_independent_set(N, pair_to_z, list(range(1, N + 1)), deadline)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    # Restart 3: random order
    if time.time() < deadline - 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = _greedy_independent_set(N, pair_to_z, order, deadline)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
    
    # Restart 4: another random order
    if time.time() < deadline - 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S4 = _greedy_independent_set(N, pair_to_z, order, deadline)
        if len(S4) > best_size:
            best_S = S4
            best_size = len(S4)
    
    # Local search on the best solution found
    if time.time() < deadline - 2:
        best_S = _local_search(best_S, N, pair_to_z, deadline)
    
    return list(best_S)
