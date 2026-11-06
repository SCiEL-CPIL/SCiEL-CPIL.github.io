import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    
    # Precompute prime factorizations for faster lcm computation
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Precompute factorizations for all numbers
    factorizations = [None] * (N + 1)
    for i in range(1, N + 1):
        factorizations[i] = factorize(i)
    
    # Fast lcm using factorizations
    def fast_lcm(a, b):
        fa = factorizations[a]
        fb = factorizations[b]
        max_exp = {}
        for p, e in fa.items():
            max_exp[p] = max(e, fb.get(p, 0))
        for p, e in fb.items():
            if p not in max_exp:
                max_exp[p] = e
        result = 1
        for p, e in max_exp.items():
            result *= p ** e
        return result
    
    # Precompute compatibility bitmask
    compat_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if fast_lcm(i, j) <= N:
                mask |= (1 << (j - 1))
        compat_mask[i] = mask
    
    def time_up():
        return time.time() > deadline
    
    # Strategy 1: Find best divisor set
    best_divisor_set = []
    max_divs = 0
    for M in range(1, N + 1):
        divs = []
        for d in range(1, int(M**0.5) + 1):
            if M % d == 0:
                divs.append(d)
                if d != M // d:
                    divs.append(M // d)
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_divisor_set = divs
    
    best_clique = list(best_divisor_set)
    
    # Strategy 2: Greedy expansion from best divisor set
    def greedy_expand(initial_set):
        clique = list(initial_set)
        in_clique = set(clique)
        # Try adding vertices in order of decreasing degree
        candidates = [v for v in range(1, N + 1) if v not in in_clique]
        candidates.sort(key=lambda v: -bin(compat_mask[v]).count('1'))
        
        for v in candidates:
            if time_up():
                break
            if all(compat_mask[v] & (1 << (u - 1)) for u in clique):
                clique.append(v)
                in_clique.add(v)
        return clique
    
    expanded = greedy_expand(best_divisor_set)
    if len(expanded) > len(best_clique):
        best_clique = expanded
    
    # Strategy 3: Standard greedy with multiple orders
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(compat_mask[i]).count('1')
    
    sorted_vertices = sorted(range(1, N + 1), key=lambda v: -degrees[v])
    
    def greedy_clique(order):
        clique = []
        for v in order:
            if time_up():
                break
            if all(compat_mask[v] & (1 << (u - 1)) for u in clique):
                clique.append(v)
        return clique
    
    orders = [
        sorted_vertices,
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda v: -degrees[v] / v),
        sorted(range(1, N + 1), key=lambda v: v),
    ]
    
    for order in orders:
        if time_up():
            break
        g_clique = greedy_clique(order)
        if len(g_clique) > len(best_clique):
            best_clique = g_clique
    
    # Strategy 4: Local search with swaps
    import random
    random.seed(42)
    
    def local_search(clique, iterations=100):
        clique = list(clique)
        in_clique = set(clique)
        improved = True
        it = 0
        while improved and it < iterations and not time_up():
            improved = False
            it += 1
            # Try adding any compatible vertex
            for v in range(1, N + 1):
                if v not in in_clique and all(compat_mask[v] & (1 << (u - 1)) for u in clique):
                    clique.append(v)
                    in_clique.add(v)
                    improved = True
                    break
            if improved:
                continue
            # Try 2-swap: remove u, add v
            for u in list(clique):
                for v in range(1, N + 1):
                    if v in in_clique:
                        continue
                    rest = [x for x in clique if x != u]
                    if all(compat_mask[v] & (1 << (w - 1)) for w in rest):
                        # Check if v is compatible with u (it's not, since we're swapping)
                        # Actually we need v compatible with all of rest
                        # This gives same size, but may lead to larger later
                        new_clique = rest + [v]
                        # Try to expand from new_clique
                        new_in = set(new_clique)
                        added = False
                        for w in range(1, N + 1):
                            if w not in new_in and all(compat_mask[w] & (1 << (x - 1)) for x in new_clique):
                                new_clique.append(w)
                                new_in.add(w)
                                added = True
                        if len(new_clique) > len(clique):
                            clique = new_clique
                            in_clique = new_in
                            improved = True
                            break
                if improved:
                    break
        return clique
    
    ls_clique = local_search(best_clique, iterations=50)
    if len(ls_clique) > len(best_clique):
        best_clique = ls_clique
    
    # Strategy 5: Random restarts with perturbation
    for restart in range(100):
        if time_up():
            break
        
        # Start from a random subset of the best clique plus random vertices
        base = list(best_clique)
        random.shuffle(base)
        # Keep a random fraction
        keep_count = max(1, len(base) // 2)
        seed_clique = base[:keep_count]
        
        # Add some random vertices
        extra = random.sample([v for v in range(1, N + 1) if v not in seed_clique], min(5, N - len(seed_clique)))
        seed_clique.extend(extra)
        
        # Greedy expand
        g_clique = greedy_expand(seed_clique)
        if len(g_clique) > len(best_clique):
            best_clique = g_clique
        
        # Try local search on result
        if len(g_clique) >= len(best_clique) - 1:
            ls = local_search(g_clique, iterations=20)
            if len(ls) > len(best_clique):
                best_clique = ls
    
    # Final: try to expand best_clique one more time
    final = greedy_expand(best_clique)
    if len(final) > len(best_clique):
        best_clique = final
    
    return sorted(best_clique)
