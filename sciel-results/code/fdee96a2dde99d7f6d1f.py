def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 75.0

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

    # Precompute edge lists for fast balance check
    bot_cycle_edges = []
    top_cycle_edges = []
    ladder_cycle_edges = [] # list of lists of edge indices

    for i in range(m):
        u, v = i, (i + 1) % m
        bot_cycle_edges.append(idx[key(u, v)])
        
        u, v = m + i, m + (i + 1) % m
        top_cycle_edges.append(idx[key(u, v)])

    for i in range(m - 1):
        ec = []
        # Bottom i
        u, v = i, i + 1
        ec.append(idx[key(u, v)])
        # Top i
        u, v = m + i, m + i + 1
        ec.append(idx[key(u, v)])
        # Matching i
        u, v = i, m + i
        ec.append(idx[key(u, v)])
        # Matching i+1
        u, v = i + 1, m + i + 1
        ec.append(idx[key(u, v)])
        ladder_cycle_edges.append(ec)

    def is_balanced_fast(sgn):
        # Bottom
        prod = 1
        for ei in bot_cycle_edges:
            prod *= sgn[ei]
        if prod == -1:
            return False
            
        # Top
        prod = 1
        for ei in top_cycle_edges:
            prod *= sgn[ei]
        if prod == -1:
            return False
            
        # Ladders
        for ec in ladder_cycle_edges:
            prod = 1
            for ei in ec:
                prod *= sgn[ei]
            if prod == -1:
                return False
                
        return True

    def spectral_radius(sgn):
        A = np.zeros((n, n))
        for (u, v) in edges:
            s = sgn[idx[key(u, v)]]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')

    def consider(sgn):
        nonlocal best_signs, best_radius
        if is_balanced_fast(sgn):
            return
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # Brute force for small m
    if 2 ** E <= 50000:
        for mask in range(2 ** E):
            if time.time() > deadline - 5:
                break
            sgn = np.ones(E, dtype=np.int8)
            for i in range(E):
                if mask & (1 << i):
                    sgn[i] = -1
            consider(sgn)
        if best_signs is not None:
            return np.asarray(best_signs, dtype=np.int8)

    # Multi-start local search
    rng = np.random.default_rng(3)
    
    # Initial seed: try a few structured ones
    seeds = []
    # All +1
    seeds.append(np.ones(E, dtype=np.int8))
    # All -1
    seeds.append(-np.ones(E, dtype=np.int8))
    # Bottom -1, Top +1, Match +1
    s1 = np.ones(E, dtype=np.int8)
    for ei in bot_cycle_edges:
        s1[ei] = -1
    seeds.append(s1)
    
    # Random seeds
    for _ in range(10):
        seeds.append(rng.choice([-1, 1], size=E).astype(np.int8))

    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        cur = seed.copy()
        # Ensure unbalanced
        if is_balanced_fast(cur):
            # Flip a random edge
            cur[rng.integers(E)] *= -1
            if is_balanced_fast(cur):
                # Try another
                cur[rng.integers(E)] *= -1
        
        cur_r = float('inf')
        if not is_balanced_fast(cur):
            cur_r = spectral_radius(cur)
            if cur_r < best_radius:
                best_radius = cur_r
                best_signs = cur.copy()
        
        # Local search
        improved = True
        while improved:
            improved = False
            # Randomize order? Or fixed? Fixed is faster, might get stuck.
            # Let's use fixed order for speed, but maybe shuffle occasionally?
            # For now, fixed order 0..E-1
            for ei in range(E):
                if time.time() > deadline - 5:
                    break
                
                cur[ei] *= -1
                if is_balanced_fast(cur):
                    cur[ei] *= -1
                    continue
                
                r = spectral_radius(cur)
                if r < cur_r - 1e-9:
                    cur_r = r
                    improved = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur.copy()
                else:
                    cur[ei] *= -1
            
            if time.time() > deadline - 5:
                break

    if best_signs is None:
        # Fallback
        best_signs = np.ones(E, dtype=np.int8)
        if is_balanced_fast(best_signs):
            best_signs[0] = -1

    return np.asarray(best_signs, dtype=np.int8)
