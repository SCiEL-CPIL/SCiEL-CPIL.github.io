import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initialize with rank-matching heuristic
    # Rows of a with largest totals -> Rows of b with smallest totals
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # Cost = sum(a[i,j] * b[p[i], p[j]])
    # Vectorized: np.sum(a * b[np.ix_(p, p)])
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # SA Parameters
    # For n=100, typical cost scale is around 10^5 to 10^6.
    # Initial temperature should be high enough to accept bad moves.
    # Let's estimate a typical delta magnitude.
    # Heuristic: T0 = 1000 * max(a.sum()) / n or similar.
    # Or just start high and cool down.
    T = 100000.0 
    T_min = 1.0
    alpha = 0.99999  # Cooling rate, needs to be tuned to reach T_min in ~100k-1000k steps
    
    # Estimate steps per second to adjust alpha? 
    # Let's just run until time limit.
    
    # Pre-allocate arrays for speed if needed, but n is small.
    
    while time.time() - start_time < time_limit:
        # Pick random swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Calculate delta cost for swapping x and y at positions i and j
        # Delta = sum_{k != i, j} [ (a[x,k] - a[y,k]) * (b[x, p[k]] - b[y, p[k]]) + (a[k,x] - a[k,y]) * (b[p[k], x] - b[p[k], y]) ] 
        #       + (a[y,x] - a[x,y]) * (b[x, y] - b[y, x])
        
        # Mask for k != i, j
        # Creating a mask and indexing is O(n). 
        # Since n=100, this is very fast.
        
        # Optimization: Instead of masking, we can compute the full sum and subtract the i,j terms?
        # Full sum S = sum_k [ ... ]
        # The formula above is for k != i, j.
        # Let's stick to the explicit mask for clarity and correctness, as n is small.
        
        # Vectorized calculation
        # We need p[ks] where ks are indices excluding i and j
        # p is a 1D array.
        
        # Create mask
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            # p[ks] is an array of locations for facilities at indices ks
            pk = p[ks]
            
            # Term 1: (a[x, ks] - a[y, ks]) * (b[x, pk] - b[y, pk])
            # a[x, ks] is a vector of length n-2
            # b[x, pk] is a vector of length n-2
            
            t1 = (a[x, ks] - a[y, ks]) * (b[x, pk] - b[y, pk])
            # Term 2: (a[ks, x] - a[ks, y]) * (b[pk, x] - b[pk, y])
            t2 = (a[ks, x] - a[ks, y]) * (b[pk, x] - b[pk, y])
            
            delta = np.sum(t1 + t2)
        else:
            delta = 0.0
            
        # Correction for the pair (i, j) and (j, i)
        # Old: a[x,y]b[x,y] + a[y,x]b[y,x]
        # New: a[x,y]b[y,x] + a[y,x]b[x,y]
        # Diff: a[x,y](b[y,x] - b[x,y]) + a[y,x](b[x,y] - b[y,x])
        #      = (a[y,x] - a[x,y]) * (b[x,y] - b[y,x])
        delta += (a[y, x] - a[x, y]) * (b[x, y] - b[y, x])
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            # Avoid log(0) or overflow
            if T > 0:
                prob = np.exp(-delta / T)
                accept = (np.random.random() < prob)
            else:
                accept = False
        
        if accept:
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_p.tolist()
