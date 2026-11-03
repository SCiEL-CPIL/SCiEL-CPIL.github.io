# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    # p[i] = location of facility i
    # Use float64 to avoid overflow
    B_perm = b[np.arange(len(p)), p]  # shape (n, n): B_perm[i, j] = b[p[i], p[j]]
    return float(np.sum(a.astype(np.float64) * B_perm))


def _greedy_init(a, b, n, rng=None):
    """Greedy construction: place facilities in order of decreasing total flow.
    Returns p where p[i] = location of facility i."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Total flow of facility i
    flow_i = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-flow_i, kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Current cost contribution tracking
    # When placing facility i at location loc, the incremental cost is:
    # sum_j a[i,j]*b[loc, p[j]] + sum_j a[j,i]*b[p[j], loc] + a[i,i]*b[loc,loc]
    
    # Precompute for each location loc: b[loc, :] and b[:, loc]
    # We'll maintain arrays for placed facilities
    
    placed = []  # list of (facility, location)
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used[loc]:
                continue
            # Compute incremental cost of placing facility idx at loc
            inc = 0.0
            # a[idx, idx] * b[loc, loc]
            inc += a[idx, idx] * b[loc, loc]
            for k, (f, l) in enumerate(placed):
                inc += a[idx, f] * b[loc, l] + a[f, idx] * b[l, loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        if best_loc == -1:
            best_loc = int(np.where(~used)[0][0])
        p[idx] = best_loc
        used[best_loc] = True
        placed.append((idx, best_loc))
    
    return p


def _random_init(n, rng):
    """Random permutation initialization."""
    p = rng.permutation(n)
    return p.astype(np.int64)


def _local_search_2opt(a, b, p, deadline, max_iters=50):
    """2-opt local search. p[i] = location of facility i.
    Tries swapping locations of two facilities."""
    n = len(p)
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    cur_cost = _cost(a_f, b_f, p)
    p = p.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Precompute row sums and column sums of a for efficiency
        a_row_sums = a_f.sum(axis=1)  # sum_j a[i,j]
        a_col_sums = a_f.sum(axis=0)  # sum_j a[j,i]
        
        # For 2-opt: swap facilities at positions i and j in the permutation
        # p[i] and p[j] are locations. Swapping means p[i] <-> p[j]
        # i.e., facility i goes to location p[j], facility j goes to location p[i]
        
        for i in range(n):
            if time.time() > deadline:
                break
            loc_i = p[i]
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                loc_j = p[j]
                
                # Compute delta for swapping p[i] and p[j]
                # Old: a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i] + a[i,i]*b[loc_i,loc_i] + a[j,j]*b[loc_j,loc_j]
                # New: a[i,j]*b[loc_j, loc_i] + a[j,i]*b[loc_i, loc_j] + a[i,i]*b[loc_j,loc_j] + a[j,j]*b[loc_i,loc_i]
                # Plus cross terms with all other facilities k != i, j
                # Old cross: a[i,k]*b[loc_i, p[k]] + a[k,i]*b[p[k], loc_i] + a[j,k]*b[loc_j, p[k]] + a[k,j]*b[p[k], loc_j]
                # New cross: a[i,k]*b[loc_j, p[k]] + a[k,i]*b[p[k], loc_j] + a[j,k]*b[loc_i, p[k]] + a[k,j]*b[p[k], loc_i]
                
                # Delta = New - Old
                # For i,i term: a[i,i]*(b[loc_j,loc_j] - b[loc_i,loc_i])
                # For j,j term: a[j,j]*(b[loc_i,loc_i] - b[loc_j,loc_j])
                # For i,j term: a[i,j]*(b[loc_j,loc_i] - b[loc_i,loc_j]) + a[j,i]*(b[loc_i,loc_j] - b[loc_j,loc_i])
                # = (a[i,j] - a[j,i]) * (b[loc_j,loc_i] - b[loc_i,loc_j])
                
                di_i = a_f[i, i] * (b_f[loc_j, loc_j] - b_f[loc_i, loc_i])
                di_j = a_f[j, j] * (b_f[loc_i, loc_i] - b_f[loc_j, loc_j])
                di_ij = (a_f[i, j] - a_f[j, i]) * (b_f[loc_j, loc_i] - b_f[loc_i, loc_j])
                
                # Cross terms: sum over k != i, j
                # (a[i,k] - a[j,k]) * (b[loc_j, p[k]] - b[loc_i, p[k]]) 
                # + (a[k,i] - a[k,j]) * (b[p[k], loc_j] - b[p[k], loc_i])
                
                # Vectorized: 
                # b[loc_j, p] - b[loc_i, p] = b_f[loc_j, p] - b_f[loc_i, p]
                # b[p, loc_j] - b[p, loc_i] = b_f[p, loc_j] - b_f[p, loc_i]
                
                delta = di_i + di_j + di_ij
                delta += np.dot(a_f[i, :] - a_f[j, :], b_f[loc_j, p] - b_f[loc_i, p])
                delta += np.dot(a_f[:, i] - a_f[:, j], b_f[p, loc_j] - b_f[p, loc_i])
                
                # Subtract the i and j terms that were included in the dot products
                # In the first dot: k=i and k=j terms
                # k=i: (a[i,i]-a[j,i]) * (b[loc_j, loc_i] - b[loc_i, loc_i])
                # k=j: (a[i,j]-a[j,j]) * (b[loc_j, loc_j] - b[loc_i, loc_j])
                # In the second dot: k=i and k=j terms
                # k=i: (a[i,i]-a[i,j]) * (b[loc_i, loc_j] - b[loc_i, loc_i])
                # k=j: (a[j,i]-a[j,j]) * (b[loc_j, loc_j] - b[loc_j, loc_i])
                
                sub1_i = (a_f[i, i] - a_f[j, i]) * (b_f[loc_j, loc_i] - b_f[loc_i, loc_i])
                sub1_j = (a_f[i, j] - a_f[j, j]) * (b_f[loc_j, loc_j] - b_f[loc_i, loc_j])
                sub2_i = (a_f[i, i] - a_f[i, j]) * (b_f[loc_i, loc_j] - b_f[loc_i, loc_i])
                sub2_j = (a_f[j, i] - a_f[j, j]) * (b_f[loc_j, loc_j] - b_f[loc_j, loc_i])
                
                delta -= sub1_i + sub1_j + sub2_i + sub2_j
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cur_cost += delta
                    improved = True
        
        if not improved:
            break
    
    return p, cur_cost


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    initializations = []
    
    # 1. Greedy init
    p_greedy = _greedy_init(a, b, n, rng)
    initializations.append(p_greedy)
    
    # 2. Random inits
    for _ in range(5):
        p_rand = _random_init(n, rng)
        initializations.append(p_rand)
    
    # 3. Greedy with different ordering (by row sum only)
    flow_row = a_f.sum(axis=1)
    order2 = np.argsort(-flow_row, kind='stable')
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    placed = []
    for idx in order2:
        best_loc = -1
        best_cost_loc = np.inf
        for loc in range(n):
            if used[loc]:
                continue
            inc = float(a[idx, idx] * b[loc, loc])
            for f, l in placed:
                inc += float(a[idx, f] * b[loc, l] + a[f, idx] * b[l, loc])
            if inc < best_cost_loc:
                best_cost_loc = inc
                best_loc = loc
        if best_loc == -1:
            best_loc = int(np.where(~used)[0][0])
        p[idx] = best_loc
        used[best_loc] = True
        placed.append((idx, best_loc))
    initializations.append(p)
    
    # Run local search on each initialization
    for init_p in initializations:
        if time.time() > deadline:
            break
        
        # Quick cost check
        c = _cost(a_f, b_f, init_p)
        
        # Run local search
        ls_p, ls_cost = _local_search_2opt(a_f, b_f, init_p, deadline)
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_p = ls_p.copy()
    
    # ILS: perturb the best solution and re-local-search
    if best_p is not None:
        for _ in range(200):  # Max ILS iterations, but time-limited
            if time.time() > deadline:
                break
            
            # Perturb: random 2-swap or 3-swap
            if rng.random() < 0.5:
                # 2-swap
                i, j = rng.choice(n, size=2, replace=False)
                perturbed = best_p.copy()
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            else:
                # 3-swap (cycle)
                i, j, k = rng.choice(n, size=3, replace=False)
                perturbed = best_p.copy()
                perturbed[i], perturbed[j], perturbed[k] = perturbed[j], perturbed[k], perturbed[i]
            
            # Local search from perturbed
            ls_p, ls_cost = _local_search_2opt(a_f, b_f, perturbed, deadline)
            
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_p = ls_p.copy()
    
    if best_p is None:
        # Fallback: return identity
        return list(range(n))
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
