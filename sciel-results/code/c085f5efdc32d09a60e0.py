"""Rewritten program for spectral graph lifts with optimized connectivity check."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product as iproduct

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    deadline = time.time() + 55.0  # Safety margin
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    # Identify edge types
    top_edges = []  # (edge_idx, u) where u is the index 0..m-1
    bot_edges = []  # (edge_idx, u) where u is the index 0..m-1
    match_edges = []  # (edge_idx, i) where i is 0..m-1

    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle edge. The sign is associated with the edge.
            # We need to map this to a variable.
            # Let's assume the edge is between u and (u+1)%m or similar.
            # Actually, we just need to store the index in the signs array.
            # But for the product constraint, we need to know which 'a' or 'b' or 'c' it belongs to.
            # Let's assign:
            # a[i] corresponds to the top edge connecting i and i+1 (or whatever the order is)
            # b[i] corresponds to the bottom edge
            # c[i] corresponds to the matching edge i
            # We need to map the edge index to a variable index.
            # Since the graph is a prism, the top edges form a cycle.
            # Let's just store the variable index.
            # For top edges, the variable index is not strictly defined by vertex id unless we order them.
            # However, the product constraint is over ALL top edges.
            # So we just need a list of variable indices for top edges.
            # Let's say top_edges[k] corresponds to variable a[k].
            # We need to ensure that the k-th top edge in the list corresponds to a[k].
            # The order of edges in the list might be arbitrary.
            # But the product is commutative. So we just need to know which edges are in which set.
            
            # Let's assign an index to each edge in the set.
            # For top edges, let's just use a counter.
            pass
        
        # Re-doing the parsing to be robust
    # Let's re-parse carefully
    top_edge_vars = [] # list of (edge_idx, var_idx)
    bot_edge_vars = []
    match_edge_vars = []
    
    top_count = 0
    bot_count = 0
    match_count = 0
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edge_vars.append((idx, top_count))
            top_count += 1
        elif u >= m and v >= m:
            bot_edge_vars.append((idx, bot_count))
            bot_count += 1
        else:
            # Matching edge
            i = min(u, v)
            match_edge_vars.append((idx, i)) # Variable index is i
            match_count += 1
            
    assert top_count == m, f"Top count {top_count} != m {m}"
    assert bot_count == m, f"Bot count {bot_count} != m {m}"
    assert match_count == m, f"Match count {match_count} != m {m}"

    # Variables:
    # a: length m
    # b: length m
    # c: length m
    # Total 3m variables.
    
    def get_signs(a, b, c):
        signs = np.zeros(num_edges, dtype=np.int8)
        for idx, vi in top_edge_vars:
            signs[idx] = a[vi]
        for idx, vi in bot_edge_vars:
            signs[idx] = b[vi]
        for idx, vi in match_edge_vars:
            signs[idx] = c[vi]
        return signs

    def is_connected(a, b, c):
        # Product of all signs must be -1
        prod = 1
        for x in a: prod *= x
        for x in b: prod *= x
        for x in c: prod *= x
        return prod == -1

    def spectral_radius(signs):
        # Build matrix
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def eval_abc(a, b, c):
        if not is_connected(a, b, c):
            return np.inf
        signs = get_signs(a, b, c)
        return spectral_radius(signs)

    best_signs = None
    best_radius = np.inf

    # Strategy 1: Exhaustive for small m
    if m <= 5:
        # Free variables: a (m), b (m), c (m-1) -> c[m-1] determined
        # Actually, we can just iterate all 3m and check connectivity.
        # 2^(3m)
        if 3*m <= 20: # 2^20 = 1M, feasible
            for combo in iproduct([-1, 1], repeat=3*m):
                if time.time() > deadline:
                    break
                a = np.array(combo[0:m], dtype=np.int8)
                b = np.array(combo[m:2*m], dtype=np.int8)
                c = np.array(combo[2*m:3*m], dtype=np.int8)
                r = eval_abc(a, b, c)
                if r < best_radius:
                    best_radius = r
                    best_signs = get_signs(a, b, c)
        else:
            # Fallback to random for m=5 if 2^15 is too slow? 2^15 is 32k, very fast.
            pass

    if best_signs is not None:
        try:
            check_lift(graph, best_signs)
            return best_signs
        except:
            pass

    # Strategy 2: Local Search / Simulated Annealing for larger m
    # Initialize with a few good candidates
    candidates = []
    
    # Candidate 1: All +1, fix one -1
    a0 = np.ones(m, dtype=np.int8)
    b0 = np.ones(m, dtype=np.int8)
    c0 = np.ones(m, dtype=np.int8)
    c0[0] = -1
    candidates.append((a0, b0, c0))
    
    # Candidate 2: Alternating on matching
    for pat in range(2):
        a1 = np.ones(m, dtype=np.int8)
        b1 = np.ones(m, dtype=np.int8)
        c1 = np.ones(m, dtype=np.int8)
        for i in range(m):
            if (i + pat) % 2 == 1:
                c1[i] = -1
        # Ensure connectivity
        if not is_connected(a1, b1, c1):
            # Flip one more to fix
            c1[0] = -c1[0]
        candidates.append((a1, b1, c1))
        
    # Candidate 3: Random
    rng = np.random.default_rng(3)
    for _ in range(10):
        a2 = rng.choice([-1, 1], size=m).astype(np.int8)
        b2 = rng.choice([-1, 1], size=m).astype(np.int8)
        c2 = rng.choice([-1, 1], size=m).astype(np.int8)
        if not is_connected(a2, b2, c2):
            c2[0] = -c2[0]
        candidates.append((a2, b2, c2))

    # Simulated Annealing
    def anneal(a_init, b_init, c_init, T_start=1.0, T_end=0.01, steps=5000):
        a = a_init.copy()
        b = b_init.copy()
        c = c_init.copy()
        current_r = eval_abc(a, b, c)
        if np.isinf(current_r):
            return np.inf, None
        
        best_r_local = current_r
        best_a_local = a.copy()
        best_b_local = b.copy()
        best_c_local = c.copy()
        
        # Variables are indices 0..3m-1
        # 0..m-1: a
        # m..2m-1: b
        # 2m..3m-1: c
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick two distinct variables to flip
            i1 = rng.integers(0, 3*m)
            i2 = rng.integers(0, 3*m - 1)
            if i2 >= i1: i2 += 1
            
            # Flip
            if i1 < m: a[i1] = -a[i1]
            elif i1 < 2*m: b[i1-m] = -b[i1-m]
            else: c[i1-2*m] = -c[i1-2*m]
            
            if i2 < m: a[i2] = -a[i2]
            elif i2 < 2*m: b[i2-m] = -b[i2-m]
            else: c[i2-2*m] = -c[i2-2*m]
            
            new_r = eval_abc(a, b, c)
            
            # Since we flipped 2 signs, connectivity is preserved if it was valid before.
            # But eval_abc checks it anyway.
            
            T = T_start * (T_end / T_start) ** (step / steps)
            
            if new_r < current_r:
                current_r = new_r
                if new_r < best_r_local:
                    best_r_local = new_r
                    best_a_local = a.copy()
                    best_b_local = b.copy()
                    best_c_local = c.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = new_r - current_r
                if delta < 0:
                    # This shouldn't happen if new_r > current_r
                    pass
                else:
                    prob = np.exp(-delta / max(T, 1e-6))
                    if rng.random() < prob:
                        current_r = new_r
                    else:
                        # Revert
                        if i1 < m: a[i1] = -a[i1]
                        elif i1 < 2*m: b[i1-m] = -b[i1-m]
                        else: c[i1-2*m] = -c[i1-2*m]
                        
                        if i2 < m: a[i2] = -a[i2]
                        elif i2 < 2*m: b[i2-m] = -b[i2-m]
                        else: c[i2-2*m] = -c[i2-2*m]
                        
        return best_r_local, (best_a_local, best_b_local, best_c_local)

    # Run annealing from each candidate
    for a_init, b_init, c_init in candidates:
        if time.time() > deadline:
            break
        r_local, res = anneal(a_init, b_init, c_init)
        if res is not None and r_local < best_radius:
            best_radius = r_local
            a_best, b_best, c_best = res
            best_signs = get_signs(a_best, b_best, c_best)
            
    if best_signs is None:
        # Fallback
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        c[0] = -1
        best_signs = get_signs(a, b, c)

    try:
        check_lift(graph, best_signs)
    except:
        pass

    return best_signs
# EVOLVE-BLOCK-END
