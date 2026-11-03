"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def total_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute row sums for greedy construction
    # Use combined row+col sums for better greedy ordering
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    a_combined = a_row_sums + a_col_sums
    
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    b_combined = b_row_sums + b_col_sums
    
    # Greedy construction: assign facilities with largest combined sums 
    # to locations with smallest combined sums
    heavy_order = np.argsort(-a_combined, kind="stable")
    central_order = np.argsort(b_combined, kind="stable")
    
    # Build initial assignment
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Compute initial cost
    current_cost = total_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for fast delta computation
    # For 2-opt swap of positions i and j:
    # Delta = sum over k != i,j of a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]]) 
    #        + sum over k != i,j of a[j,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #        + a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  [but b is symmetric so this is 0]
    #        + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])  [same, 0 if symmetric]
    # Simplified for symmetric b:
    # Delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
    #        + (a[i,j] - a[j,i]) * (b[p[i],p[j]] - b[p[j],p[i]])
    # If both a and b are symmetric: a[i,j]=a[j,i], b[p[i],p[j]]=b[p[j],p[i]]
    # So Delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
    
    # Check if matrices are symmetric
    a_sym = np.allclose(a, a.T)
    b_sym = np.allclose(b, b.T)
    
    # 2-opt local search
    def delta_2opt(i, j, p, a, b, a_sym, b_sym):
        """Compute cost delta for swapping positions i and j."""
        pi = p[i]
        pj = p[j]
        
        if a_sym and b_sym:
            # Simplified formula
            # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pi,p[k]] - b[pj,p[k]])
            ai = a[i]
            aj = a[j]
            bpi = b[pi]
            bpj = b[pj]
            
            # Vectorized computation excluding i and j
            diff_a = ai - aj
            diff_b = bpi - bpj
            delta = np.sum(diff_a * diff_b)
            # Subtract the terms where k=i and k=j (they were included)
            # k=i: (a[i,i]-a[j,i]) * (b[pi,pi] - b[pj,pi])
            # k=j: (a[i,j]-a[j,j]) * (b[pi,pj] - b[pj,pj])
            delta -= (a[i,i] - a[j,i]) * (b[pi,pi] - b[pj,pi])
            delta -= (a[i,j] - a[j,j]) * (b[pi,pj] - b[pj,pj])
            return int(delta)
        else:
            # General case
            pi = p[i]
            pj = p[j]
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = p[k]
                delta += a[i,k] * (b[pi,pk] - b[pj,pk])
                delta += a[j,k] * (b[pj,pk] - b[pi,pk])
            # i-j and j-i terms
            delta += a[i,j] * (b[pi,pj] - b[pj,pi])
            delta += a[j,i] * (b[pj,pi] - b[pi,pj])
            return int(delta)
    
    # Local search phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                d = delta_2opt(i, j, p, a, b, a_sym, b_sym)
                if d < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    # If we have time left, try simulated annealing or more restarts
    if time.time() < deadline - 5.0:
        # Try a few restarts with perturbation
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution
            p = best_p.copy()
            # Random swap
            idx = np.random.choice(n, 2, replace=False)
            p[idx[0]], p[idx[1]] = p[idx[1]], p[idx[0]]
            current_cost = total_cost(a, b, p)
            
            # Local search from perturbed solution
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5.0:
                            break
                        d = delta_2opt(i, j, p, a, b, a_sym, b_sym)
                        if d < 0:
                            p[i], p[j] = p[j], p[i]
                            current_cost += d
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
