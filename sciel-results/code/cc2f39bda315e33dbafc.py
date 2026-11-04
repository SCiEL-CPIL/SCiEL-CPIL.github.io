# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def compute_delta(p, i, j):
        """Compute cost change for swapping positions i and j in permutation p."""
        u, v = p[i], p[j]
        # delta = sum_{k != i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
        #        + sum_{k != i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        #        + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        #        - (a[i,i]-a[j,i])*(b[v,u]-b[u,u])  [k=i term]
        #        - (a[i,j]-a[j,j])*(b[v,v]-b[u,v])  [k=j term]
        #        - (a[i,i]-a[i,j])*(b[u,v]-b[u,u])  [k=i term for second sum]
        #        - (a[j,i]-a[j,j])*(b[v,v]-b[v,u])  [k=j term for second sum]
        
        D1 = a[i, :] - a[j, :]
        D2 = a[:, i] - a[:, j]
        
        diff1 = b[v, p] - b[u, p]
        diff2 = b[p, v] - b[p, u]
        
        Sum1_all = D1 @ diff1
        Sum2_all = D2 @ diff2
        
        Sub1_i = D1[i] * (b[v, u] - b[u, u])
        Sub2_i = D2[i] * (b[u, v] - b[u, u])
        Sub1_j = D1[j] * (b[v, v] - b[u, v])
        Sub2_j = D2[j] * (b[v, v] - b[v, u])
        
        Sum1 = Sum1_all - Sub1_i - Sub1_j
        Sum2 = Sum2_all - Sub2_i - Sub2_j
        
        Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        return Cross + Sum1 + Sum2

    # 1. Multiple initial solutions: rank matching + random restarts
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Rank matching init
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order
    
    best_cost = compute_cost(p_init)
    best_p = p_init.copy()
    
    # Try a few random inits with quick local search
    rng_init = np.random.RandomState(42)
    for _ in range(5):
        p_rand = rng_init.permutation(n).astype(np.int64)
        cost_rand = compute_cost(p_rand)
        if cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand.copy()
    
    # 2. Fast local search with vectorized delta computation
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute for faster delta computation
    # We'll use a smarter approach: precompute row/col sums to speed up delta
    
    improved = True
    local_search_deadline = deadline - 40.0  # Reserve 40s for SA
    while improved and time.time() < local_search_deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > local_search_deadline:
                break
            u = p[i]
            for j in range(i + 1, n):
                delta = compute_delta(p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < -1e-10:
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break
    
    # 3. Simulated Annealing with slower cooling
    sa_deadline = deadline - 2.0
    if time.time() < sa_deadline:
        p = best_p.copy()
        current_cost = best_cost
        T_start = max(10.0, abs(best_cost) * 0.005)
        T_end = 0.01
        cooling_rate = 0.9995
        
        T = T_start
        rng = random.Random(42)
        
        while T > T_end and time.time() < sa_deadline:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            
            delta = compute_delta(p, i, j)
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= cooling_rate
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
