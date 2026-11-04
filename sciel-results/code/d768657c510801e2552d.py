"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    
    # Helper to compute full cost
    def compute_cost(p):
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
        
    # Helper to compute delta cost for swapping i and j
    # p is the current permutation array
    def delta_cost(i, j, p):
        u = p[i]
        v = p[j]
        
        # We need to sum over k != i, j
        # delta = sum_k [ a[i,k](b[v,k'] - b[u,k']) + a[k,i](b[k',v] - b[k',u]) + a[j,k](b[u,k'] - b[v,k']) + a[k,j](b[k',u] - b[k',v]) ]
        # where k' = p[k]
        # Note: b indices are locations. p[k] is the location of facility k.
        
        # Let's extract rows/cols for efficiency
        # We need b[v, p[k]] and b[u, p[k]] for all k != i,j
        # This is b[v, :] and b[u, :] but we must exclude k=i,j from the sum?
        # Actually, the formula derived earlier:
        # Delta = sum_{k != i,j} [ ... ] + (a[i,j] b[v,u] + a[j,i] b[u,v]) - (a[i,j] b[u,v] + a[j,i] b[v,u])
        
        # Let's compute the sum over all k first, then subtract k=i and k=j terms if they were included?
        # It's safer to just loop or use vectorization excluding i,j.
        
        # Vectorized approach:
        # Create a mask for k != i, j
        # But indexing p[k] for all k is easy.
        
        # Term 1: a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # Term 2: a[k,i] * (b[p[k], v] - b[p[k], u])
        # Term 3: a[j,k] * (b[u, p[k]] - b[v, p[k]])
        # Term 4: a[k,j] * (b[p[k], u] - b[p[k], v])
        
        # Let's compute vectors for all k, then set i,j to 0 to exclude them.
        
        # b[v, p] is the vector of distances from location v to all assigned locations.
        # b[u, p] is the vector of distances from location u to all assigned locations.
        # b[p, v] is the vector of distances from all assigned locations to location v.
        # b[p, u] is the vector of distances from all assigned locations to location u.
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        b_p_v = b[p, v]
        b_p_u = b[p, u]
        
        # Differences
        diff_v_u_out = b_v_p - b_u_p  # b[v, p[k]] - b[u, p[k]]
        diff_v_u_in = b_p_v - b_p_u   # b[p[k], v] - b[p[k], u]
        
        # Sum 1: sum_k a[i,k] * diff_v_u_out[k]
        # Sum 2: sum_k a[k,i] * diff_v_u_in[k]
        # Sum 3: sum_k a[j,k] * (-diff_v_u_out[k])
        # Sum 4: sum_k a[k,j] * (-diff_v_u_in[k])
        
        # Note: a is symmetric? Not necessarily. b is symmetric? Usually yes for distances.
        # If b is symmetric, diff_v_u_in = -diff_v_u_out? No, b[p,k] vs b[k,p].
        # If b is symmetric, b[p[k], v] = b[v, p[k]]. So b_p_v = b_v_p.
        # And b_p_u = b_u_p.
        # So diff_v_u_in = diff_v_u_out.
        
        # Let's assume b is symmetric to simplify if possible, but code generally.
        
        s1 = np.dot(a[i, :], diff_v_u_out)
        s2 = np.dot(a[:, i], diff_v_u_in)
        s3 = np.dot(a[j, :], -diff_v_u_out)
        s4 = np.dot(a[:, j], -diff_v_u_in)
        
        # Subtract contributions from k=i and k=j because the formula was for k != i,j
        # In the dot products above, k=i and k=j are included.
        
        # For k=i:
        # s1 includes a[i,i] * (b[v, p[i]] - b[u, p[i]]) = a[i,i] * (b[v, u] - b[u, u])
        # s2 includes a[i,i] * (b[p[i], v] - b[p[i], u]) = a[i,i] * (b[u, v] - b[u, u])
        # s3 includes a[j,i] * (b[u, p[i]] - b[v, p[i]]) = a[j,i] * (b[u, u] - b[v, u])
        # s4 includes a[i,j] * (b[p[i], u] - b[p[i], v]) = a[i,j] * (b[u, u] - b[u, v])
        
        # For k=j:
        # s1 includes a[i,j] * (b[v, p[j]] - b[u, p[j]]) = a[i,j] * (b[v, v] - b[u, v])
        # s2 includes a[j,i] * (b[p[j], v] - b[p[j], u]) = a[j,i] * (b[v, v] - b[v, u])
        # s3 includes a[j,j] * (b[u, p[j]] - b[v, p[j]]) = a[j,j] * (b[u, v] - b[v, v])
        # s4 includes a[j,j] * (b[p[j], u] - b[p[j], v]) = a[j,j] * (b[v, u] - b[v, v])
        
        # This is getting messy. It's often faster to just exclude i and j in the dot product
        # by zeroing out the a rows/cols temporarily or using a mask.
        # Given n=150, a small loop or mask is fine.
        
        # Let's re-calculate with mask
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        s1 = np.dot(a[i, mask], diff_v_u_out[mask])
        s2 = np.dot(a[mask, i], diff_v_u_in[mask])
        s3 = np.dot(a[j, mask], -diff_v_u_out[mask])
        s4 = np.dot(a[mask, j], -diff_v_u_in[mask])
        
        sum_part = s1 + s2 + s3 + s4
        
        # Cross terms for i,j
        # Old: a[i,j] b[u,v] + a[j,i] b[v,u]
        # New: a[i,j] b[v,u] + a[j,i] b[u,v]
        cross_delta = a[i, j] * b[v, u] + a[j, i] * b[u, v] - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
        
        return sum_part + cross_delta

    # Initial Solution: Greedy
    # Sort facilities by row sum of a (desc)
    # Sort locations by row sum of b (asc)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Parameters
    T = 1000.0
    T_min = 0.1
    alpha = 0.9999  # Cooling rate
    
    # Precompute some things? Not strictly necessary.
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        d = delta_cost(i, j, p)
        
        if d < 0 or rng.random() < np.exp(-d / T):
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
