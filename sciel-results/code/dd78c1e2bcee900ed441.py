import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching
    # Heavy flows (high row sum in A) to short distances (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow sum descending
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance sum ascending
    central_locations = np.argsort(b_sums, kind='stable')
    
    # p[i] is location of facility i
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # 2. Compute initial cost
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This can be computed as trace(a @ b[p[:,None], p[None,:]]) or similar
    # But simpler: 
    # b_perm = b[np.ix_(p, p)]
    # cost = np.sum(a * b_perm)
    # Let's use a faster method if possible, but for n=256, this is fine.
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # 3. Simulated Annealing
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Temperature parameters
    # Estimate max delta? 
    # Max flow * Max dist.
    max_delta = np.max(a) * np.max(b) * 2 * n # Rough upper bound
    T = max_delta * 0.1
    T_min = 1e-6
    cooling_rate = 0.99999 # Adjust based on iterations
    
    # Precompute A rows for faster access? 
    # A is already a numpy array.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick two random distinct facilities
        i, j = rng.choice(n, 2, replace=False)
        
        u = p[i]
        v = p[j]
        
        # If swapping same location (impossible as p is perm) or i==j (handled by choice)
        
        # Calculate Delta
        # Delta = 2 * sum_{k != i, j} (A[i,k] - A[j,k]) * (B[v, p[k]] - B[u, p[k]])
        
        # Vector A_diff = A[i, :] - A[j, :]
        # Vector B_diff = B[v, p] - B[u, p]
        
        A_diff = a[i, :] - a[j, :]
        B_diff = b[v, p] - b[u, p]
        
        # Dot product
        dot_prod = np.dot(A_diff, B_diff)
        
        # Subtract terms for k=i and k=j
        # Term i: A_diff[i] * B_diff[i]
        # Term j: A_diff[j] * B_diff[j]
        
        # Note: B_diff[i] = B[v, p[i]] - B[u, p[i]] = B[v, u] - B[u, u]
        #       B_diff[j] = B[v, p[j]] - B[u, p[j]] = B[v, v] - B[u, v]
        
        delta = 2 * (dot_prod - A_diff[i] * B_diff[i] - A_diff[j] * B_diff[j])
        
        # Acceptance
        if delta < 0 or (T > T_min and np.random.random() < np.exp(-delta / T)):
            # Swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling_rate
        
    return best_p.tolist()
