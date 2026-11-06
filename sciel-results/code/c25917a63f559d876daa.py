import time
import math

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start = time.time()
    deadline = 85.0
    
    N = 255
    # Baseline: all multiples of 3 (size 85)
    baseline = [3 * i for i in range(1, 86)]
    
    # Precompute prime factors
    def prime_factors(n):
        pf = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf.add(d)
                n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf
    
    pf = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        pf[i] = prime_factors(i)
    
    # Eligible elements: share a factor with 255 (i.e., divisible by 3, 5, or 17)
    eligible = [x for x in range(1, N + 1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    idx = {x: i for i, x in enumerate(eligible)}
    m = len(eligible)
    
    # compat[i] = set of indices j where gcd(eligible[i], eligible[j]) > 1
    compat = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if pf[eligible[i]] & pf[eligible[j]]:
                compat[i].add(j)
                compat[j].add(i)
    
    idx_255 = idx[255]
    
    # Strategy 1: Greedy clique from 255, ordered by degree
    degrees = [len(compat[i]) for i in range(m)]
    order = sorted(range(m), key=lambda i: -degrees[i])
    
    in_clique = [False] * m
    in_clique[idx_255] = True
    clique = [255]
    
    for j in order:
        if in_clique[j]:
            continue
        ok = True
        for k in clique:
            if idx[k] not in compat[j]:
                ok = False
                break
        if ok:
            in_clique[j] = True
            clique.append(eligible[j])
    
    # Strategy 2: Start with multiples of 3, add compatible elements
    current = set(baseline)
    for x in eligible:
        if x in current:
            continue
        ix = idx[x]
        if all(idx[y] in compat[ix] for y in current):
            current.add(x)
    
    # Pick the best
    best_set = set(baseline)
    if len(clique) > len(best_set):
        best_set = set(clique)
    if len(current) > len(best_set):
        best_set = current
    
    result = sorted(best_set)
    
    # Verify
    valid = True
    if 255 not in result:
        valid = False
    else:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) <= 1:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return baseline
    
    return result
