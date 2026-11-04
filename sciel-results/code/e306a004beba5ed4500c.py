"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Greedy Rank Matching
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precomputation for fast delta calculation
    # We maintain:
    # 1. p: current permutation (facility i is at location p[i])
    # 2. inv_p: inverse permutation (location l is assigned to facility inv_p[l])
    # 3. row_sums: for each facility i, sum of flows to other facilities weighted by current distances?
    #    Actually, for swap delta, we need:
    #    Delta(i, j) = (a[i] - a[j]) dot (b[p[i]] - b[p[j]]) ... wait, that's for linear assignment.
    #    
    #    For QAP swap delta:
    #    Let u = p[i], v = p[j].
    #    Cost change when swapping i and j:
    #    Old terms involving i or j:
    #    sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u] 
    #    + sum_k a[j,k] b[v, p[k]] + sum_k a[k,j] b[p[k], v]
    #    - 2 * a[i,j] b[u,v] (since counted twice in the sums above if k=j or k=i? 
    #      Let's be precise.
    #      Total Cost = sum_{x,y} a[x,y] b[p[x], p[y]]
    #      Terms changing are those where x or y is i or j.
    #      Let S = {i, j}.
    #      Old: sum_{x in S, y not in S} a[x,y] b[p[x], p[y]] + sum_{y in S, x not in S} a[x,y] b[p[x], p[y]] + a[i,j]b[u,v] + a[j,i]b[v,u]
    #      Wait, b is symmetric? QAP usually assumes symmetric A and B. The contract says "int64 NumPy arrays".
    #      Standard QAP: min sum_{i,j} a_ij b_{pi pj}. Usually a and b are symmetric.
    #      If not symmetric, we must be careful. Let's assume general case for safety, 
    #      but most QAPLIB instances are symmetric.
    #      
    #      Let's derive the delta for general A, B.
    #      Let u = p[i], v = p[j].
    #      New u' = v, v' = u.
    #      
    #      Change in cost = 
    #      (sum_{k != i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])) +
    #      (sum_{k != i,j} a[k,i] (b[p[k], v] - b[p[k], u])) +
    #      (sum_{k != i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])) +
    #      (sum_{k != i,j} a[k,j] (b[p[k], u] - b[p[k], v])) +
    #      (a[i,j] (b[v, u] - b[u, v])) +
    #      (a[j,i] (b[u, v] - b[v, u]))
    #      
    #      If A and B are symmetric:
    #      a[i,k] = a[k,i], b[u,p[k]] = b[p[k],u]
    #      Term 1 + Term 2: sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]]) * 2?
    #      No, Term 1 is a[i,k] * delta_b. Term 2 is a[k,i] * delta_b^T.
    #      If symmetric, Term 1 == Term 2. So 2 * sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]]).
    #      Similarly for j.
    #      And the cross terms a[i,j]...
    #      
    #      Let's use a faster vectorized approach for the delta calculation.
    #      We can precompute "attractor" vectors.
    #      Let G_i = sum_{k} a[i,k] b[p[k], :] ? No, that's O(n^3) to update.
    #      
    #      Standard fast delta for QAP swap:
    #      Let u = p[i], v = p[j].
    #      delta = 2 * ( (a[i] - a[j]) dot (b[:, v] - b[:, u]) ) - 4 * (a[i,j] * (b[u,v] - 0?)) 
    #      This formula is tricky. Let's stick to a robust O(n) delta calculation per pair if possible, 
    #      or O(n^2) for all pairs if we want to pick best.
    #      
    #      For Simulated Annealing, we pick random pairs (i, j).
    #      We need to compute delta(i, j) in O(n) or O(1) ideally.
    #      
    #      O(n) delta calculation:
    #      delta = sum_{k != i, j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    #              + (a[i,j] + a[j,i]) * (b[v, u] - b[u, v])  # Wait, the cross terms
    #      
    #      Let's re-derive carefully.
    #      C = sum_{x,y} a[x,y] b[p[x], p[y]]
    #      Swap i, j. u=p[i], v=p[j].
    #      New p': p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j.
    #      
    #      C' - C = 
    #      1. Terms where x=i, y!=i:
    #         sum_{y!=i} a[i,y] (b[v, p'[y]] - b[u, p[y]])
    #         y=j: a[i,j] (b[v, u] - b[u, v])
    #         y!=i,j: a[i,y] (b[v, p[y]] - b[u, p[y]])
    #      
    #      2. Terms where x!=i, y=i:
    #         sum_{x!=i} a[x,i] (b[p'[x], v] - b[p[x], u])
    #         x=j: a[j,i] (b[u, v] - b[v, u])
    #         x!=i,j: a[x,i] (b[p[x], v] - b[p[x], u])
    #      
    #      3. Terms where x=j, y!=j:
    #         sum_{y!=j} a[j,y] (b[u, p'[y]] - b[v, p[y]])
    #         y=i: a[j,i] (b[u, v] - b[v, u])
    #         y!=i,j: a[j,y] (b[u, p[y]] - b[v, p[y]])
    #      
    #      4. Terms where x!=j, y=j:
    #         sum_{x!=j} a[x,j] (b[p'[x], u] - b[p[x], v])
    #         x=i: a[i,j] (b[v, u] - b[u, v])
    #         x!=i,j: a[x,j] (b[p[x], u] - b[p[x], v])
    #      
    #      Summing these up:
    #      Cross terms (involving i,j directly):
    #      From 1: a[i,j](b[v,u] - b[u,v])
    #      From 2: a[j,i](b[u,v] - b[v,u])
    #      From 3: a[j,i](b[u,v] - b[v,u])
    #      From 4: a[i,j](b[v,u] - b[u,v])
    #      Total Cross: (a[i,j] + a[j,i]) * (b[v,u] - b[u,v])
    #      Note: b[v,u] - b[u,v] = - (b[u,v] - b[v,u]).
    #      So if symmetric, b[v,u]=b[u,v], cross term is 0? 
    #      Wait.
    #      Term 1 (y=j): a[i,j] b[v,u] - a[i,j] b[u,v]
    #      Term 2 (x=j): a[j,i] b[u,v] - a[j,i] b[v,u]
    #      Term 3 (y=i): a[j,i] b[u,v] - a[j,i] b[v,u]
    #      Term 4 (x=i): a[i,j] b[v,u] - a[i,j] b[u,v]
    #      Sum: 2 * (a[i,j] b[v,u] - a[i,j] b[u,v] + a[j,i] b[u,v] - a[j,i] b[v,u])
    #           = 2 * ( (a[i,j] - a[j,i]) b[v,u] - (a[i,j] - a[j,i]) b[u,v] )
    #           = 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
    #      If symmetric A and B, this is 0.
    #      
    #      Non-cross terms (k != i, j):
    #      From 1: sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    #      From 2: sum_{k!=i,j} a[k,i] (b[p[k], v] - b[p[k], u])
    #      From 3: sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
    #      From 4: sum_{k!=i,j} a[k,j] (b[p[k], u] - b[p[k], v])
    #      
    #      Group by k:
    #      sum_{k!=i,j} [ (a[i,k] + a[k,i]) * (b[v, p[k]] - b[u, p[k]]) 
    #                 + (a[j,k] + a[k,j]) * (b[u, p[k]] - b[v, p[k]]) ]
    #      = sum_{k!=i,j} [ (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]]) ]
    #      
    #      So, delta = Cross + Sum_k
    #      Cross = 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
    #      Sum_k = sum_{k!=i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
    #      
    #      This is O(n) to compute for a specific pair (i,j) if we have direct access to arrays.
    #      We can vectorize the sum part.
    
    # Simulated Annealing Parameters
    T_start = 1000.0
    T_end = 1.0
    alpha = 0.999  # Cooling rate
    
    # We need to know how many iterations we can afford.
    # Each iteration is O(n) for delta calc.
    # n=150. O(150) is very fast in Python if vectorized with NumPy.
    # Let's aim for ~100,000 to 500,000 iterations.
    
    current_T = T_start
    iter_count = 0
    max_iters = 200000 # Adjust based on time
    
    # Precompute symmetric parts to simplify delta if A, B are symmetric?
    # Let's check symmetry.
    # a_sym = (a + a.T) / 2.0
    # b_sym = (b + b.T) / 2.0
    # If the instance is symmetric, we can use simpler formulas.
    # QAPLIB instances are typically symmetric.
    # Let's assume symmetry for speed, but verify with a quick check or just use the general formula.
    # The general formula above is safe.
    
    # To speed up the sum_k part:
    # Let w_k = (a[i,k] + a[k,i] - a[j,k] - a[k,j])
    # Let d_k = b[v, p[k]] - b[u, p[k]]
    # Sum = dot(w, d)
    
    # We can precompute nothing really, just use NumPy slicing.
    
    while time.time() < deadline:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Compute delta
        # Cross term
        cross = 2.0 * (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Sum term
        # We need sum over k != i, j
        # w = (a[i, :] + a[:, i] - a[j, :] - a[:, j])
        # d = (b[v, p] - b[u, p])
        # Mask out i and j
        # This creates arrays of size n.
        
        # Optimization: 
        # w_i = a[i,i] + a[i,i] - a[j,i] - a[i,j] ... wait, indices are tricky.
        # Let's just compute the full dot product and subtract the i and j terms.
        
        w_full = a[i, :] + a[:, i] - a[j, :] - a[:, j]
        d_full = b[v, p] - b[u, p]
        
        sum_full = np.dot(w_full, d_full)
        
        # Subtract k=i term
        # w_i = a[i,i] + a[i,i] - a[j,i] - a[i,j]
        # d_i = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        w_i = 2.0 * a[i, i] - a[j, i] - a[i, j]
        d_i = b[v, u] - b[u, u]
        
        # Subtract k=j term
        # w_j = a[i,j] + a[j,i] - a[j,j] - a[j,j]
        # d_j = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        w_j = a[i, j] + a[j, i] - 2.0 * a[j, j]
        d_j = b[v, v] - b[u, v]
        
        sum_k = sum_full - w_i * d_i - w_j * d_j
        
        delta = cross + sum_k
        
        # Acceptance
        if delta < 0 or (current_T > 1e-9 and np.random.random() < np.exp(-delta / current_T)):
            # Accept swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cooling
        current_T *= alpha
        iter_count += 1
        
        # Reset if stuck? 
        # Simple SA doesn't reset. But if T is too low, we might get stuck.
        # We can monitor improvement.
        
        if iter_count % 10000 == 0:
             # Check time
             if time.time() > deadline - 5:
                 break

    return best_p.tolist()
# EVOLVE-BLOCK-END
