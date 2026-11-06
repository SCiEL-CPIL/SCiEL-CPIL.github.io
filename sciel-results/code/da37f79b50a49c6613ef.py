def construct():
    import time
    from math import gcd
    import random

    deadline = time.time() + 50  # safety margin
    N = 128
    universe = list(range(1, N + 1))

    # Precompute gcd matrix
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    def is_bad_triple(a, b, c):
        return g[a][b] == g[a][c] == g[b][c]

    def can_add(S, x):
        # Check if adding x to S creates any bad triple
        # A bad triple must involve x and two elements a, b from S
        # such that g(a,b) == g(a,x) == g(b,x)
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if is_bad_triple(a, b, x):
                    return False
        return True

    def greedy_additive(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def polish(S):
        # Try to add any remaining elements that fit
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in universe:
                if x not in S and can_add(S, x):
                    S.add(x)
                    changed = True
        return S

    best = set()
    best_size = 0
    rng = random.Random(42)

    # Try several random orderings
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
        
        order = list(universe)
        rng.shuffle(order)
        
        S = greedy_additive(order)
        S = polish(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best = S
            
    # If best is empty (shouldn't happen), fallback
    if not best:
        best = {1}
        
    return sorted(best)
