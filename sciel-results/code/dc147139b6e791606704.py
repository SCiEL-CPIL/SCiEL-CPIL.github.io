import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85

    # Precompute compatibility: comp[x] is a set of y such that lcm(x,y) <= limit
    comp = [set() for _ in range(limit + 1)]
    for i in range(1, limit + 1):
        ci = comp[i]
        for j in range(i, limit + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= limit:
                ci.add(j)
                comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        return all(y in cx for y in S)

    def get_divisors(n):
        """Return all divisors of n."""
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs

    def count_divisors(n):
        """Count the number of divisors of n."""
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    def is_valid_set(S):
        """Check if all pairs in S have lcm <= limit."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                g = gcd(a, b)
                if a * b // g > limit:
                    return False
        return True

    # Strategy 1: Find the best highly composite number whose divisors form a valid set
    best = set()
    
    # Try all numbers up to limit, find those with most divisors
    for n in range(1, limit + 1):
        if time.time() > deadline - 30:
            break
        divs = get_divisors(n)
        if len(divs) > len(best):
            best = divs

    def complete(S):
        S = set(S)
        changed = True
        # Sort candidates by degree (number of compatible elements) descending to prioritize "central" nodes
        candidates = list(range(1, limit + 1))
        candidates.sort(key=lambda x: len(comp[x]), reverse=True)
        
        while changed:
            changed = False
            for x in candidates:
                if time.time() > deadline - 5:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    best = complete(best)

    # Strategy 2: Try starting from different highly composite numbers and do local search
    candidates = []
    for n in range(1, limit + 1):
        dc = count_divisors(n)
        candidates.append((dc, n))
    candidates.sort(reverse=True)
    
    # Try the top 20 candidates
    for dc, n in candidates[:20]:
        if time.time() > deadline - 10:
            break
        divs = get_divisors(n)
        S = complete(divs)
        if len(S) > len(best):
            best = S

    # Strategy 3: Local search with swaps
    # Try to remove 1 element and add 2, or remove 2 and add 3, etc.
    # Simplified: Try to replace 1 element with another if it allows adding more?
    # Better: Hill climbing on size.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S = set(best)
        # Try removing one element and completing
        for e in list(S):
            if time.time() > deadline - 5:
                break
            S_minus = S - {e}
            S2 = complete(S_minus)
            if len(S2) > len(S):
                best = S2
                improved = True
                break
        
        # Try removing two elements and completing
        if not improved:
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(S_list)):
                    if time.time() > deadline - 5:
                        break
                    e1, e2 = S_list[i], S_list[j]
                    S_minus = S - {e1, e2}
                    S2 = complete(S_minus)
                    if len(S2) > len(S):
                        best = S2
                        improved = True
                        break
                if improved:
                    break

    # Strategy 4: Random restarts
    import random
    rng = random.Random(42)
    for it in range(50):
        if time.time() > deadline - 5:
            break
        seed = {1}
        for _ in range(rng.randint(3, 10)):
            x = rng.randint(2, limit)
            if is_valid_add(seed, x):
                seed.add(x)
        S = complete(seed)
        if len(S) > len(best):
            best = S

    return sorted(best)
