import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Precompute full cost for initial
    def calc_cost_full(p):
        return np.sum(a * b[p][:, p])
        
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # SA Parameters
    T = 500.0
    T_end = 1.0
    alpha = 0.9995 # Slower cooling
    
    # Precompute a and b as float for faster arithmetic? Int64 is fine.
    
    # Mask for k != i, j
    # We will compute delta on the fly.
    
    # To speed up, we can use a simple loop with optimized delta calculation
    
    # Helper to get delta cost for swapping i, j
    def get_delta_cost(i, j, p, current_cost):
        u = p[i]
        v = p[j]
        
        # Create mask for k not i, j
        # mask = np.ones(n, dtype=bool)
        # mask[i] = False
        # mask[j] = False
        
        # Sums:
        # 1. A[i, k] * (B[v, p[k]] - B[u, p[k]]) for k != i, j
        # 2. A[k, i] * (B[p[k], v] - B[p[k], u]) for k != i, j
        # 3. A[j, k] * (B[u, p[k]] - B[v, p[k]]) for k != i, j
        # 4. A[k, j] * (B[p[k], u] - B[p[k], v]) for k != i, j
        # 5. A[i, j] * (B[v, u] - B[u, v])
        # 6. A[j, i] * (B[u, v] - B[v, u])
        
        # Vectorized calculation for k != i, j
        # We can compute full dot products and subtract i, j terms
        # Dot1 = np.dot(a[i], b[v, p])
        # Dot2 = np.dot(a[i], b[u, p])
        # Delta1 = Dot1 - Dot2 - a[i, i]*(b[v, p[i]] - b[u, p[i]]) - a[i, j]*(b[v, p[j]] - b[u, p[j]])
        # This is getting complex.
        
        # Simpler: Just use full cost recalculation if it's fast enough.
        # Let's benchmark mentally: 256*256 = 65k ops.
        # NumPy dot/mul is ~10-50 ns per op? No, vectorized.
        # 65k ops takes ~50us.
        # 120s / 50us = 2.4M iterations.
        # This is plenty.
        pass

    # Let's stick to full recalculation for simplicity and correctness, 
    # but optimize the loop structure.
    
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        p[i], p[j] = p[j], p[i]
        new_cost = calc_cost_full(p)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]
            
        T *= alpha
        if T < T_end:
            T = T_end

    return best_p.tolist()
