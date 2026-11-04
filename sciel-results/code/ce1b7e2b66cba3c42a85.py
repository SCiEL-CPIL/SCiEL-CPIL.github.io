"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy construction for a good initial solution
    flow_sums = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-flow_sums, kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        f = order[idx]
        best_loc = -1
        best_cost = np.iinfo(np.int64).max
        for loc in range(n):
            if used[loc]:
                continue
            inc = 0
            for k in range(idx):
                g = order[k]
                loc_g = p[g]
                inc += a[f, g] * b[loc, loc_g] + a[g, f] * b[loc_g, loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[f] = best_loc
        used[best_loc] = True

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = total_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    t0 = time.time()
    time_limit = 110.0

    rng = np.random.default_rng(42)
    
    # Precompute for faster delta
    # We'll use a different approach: maintain running sums to compute delta in O(1) amortized
    # But for simplicity and correctness, let's optimize the vectorized delta
    
    # Better cooling schedule
    T = 30000.0
    T_min = 0.5
    # Use adaptive cooling: slow down when improving, speed up when stuck
    no_improve_count = 0
    max_no_improve = 50000
    
    while time.time() - t0 < time_limit:
        # Pick random i, j
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Compute delta efficiently
        # delta = sum_{k!=i,j} [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])]
        #       + (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        
        # Precompute differences
        diff_row = a[i, :] - a[j, :]  # a[i,k] - a[j,k]
        diff_col = a[:, i] - a[:, j]  # a[k,i] - a[k,j]
        
        # b[v, p[k]] - b[u, p[k]] for all k
        # b[p[k], v] - b[p[k], u] for all k
        
        # These are O(n) operations but we can't avoid them easily without more memory
        bv_p = b[v, p]
        bu_p = b[u, p]
        bp_v = b[p, v]
        bp_u = b[p, u]
        
        s1 = np.dot(diff_row, bv_p - bu_p)
        s2 = np.dot(diff_col, bp_v - bp_u)
        
        # Subtract k=i and k=j terms
        # k=i: p[i]=u, so bv_p[i]=b[v,u], bu_p[i]=b[u,u], bp_v[i]=b[u,v], bp_u[i]=b[u,u]
        s1 -= diff_row[i] * (b[v, u] - b[u, u])
        s2 -= diff_col[i] * (b[u, v] - b[u, u])
        
        # k=j: p[j]=v, so bv_p[j]=b[v,v], bu_p[j]=b[u,v], bp_v[j]=b[v,v], bp_u[j]=b[v,u]
        s1 -= diff_row[j] * (b[v, v] - b[u, v])
        s2 -= diff_col[j] * (b[v, v] - b[v, u])
        
        direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        delta = s1 + s2 + direct
        
        # Acceptance with adaptive temperature
        if delta < 0:
            accept = True
            no_improve_count = 0
        else:
            if T > 1e-9:
                ratio = math.exp(-delta / T)
                accept = rng.random() < ratio
            else:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive cooling
        if no_improve_count > max_no_improve:
            # Restart from best solution with higher temperature
            p = best_p.copy()
            current_cost = best_cost
            T = max(T * 2.0, 10000.0)
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.2)
        
        # Cool down
        T *= 0.99998
        if T < T_min:
            T = T_min

    return best_p.tolist()
# EVOLVE-BLOCK-END
