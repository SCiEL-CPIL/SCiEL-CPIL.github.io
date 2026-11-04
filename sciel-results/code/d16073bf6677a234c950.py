import numpy as np
import time
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _spectral_hungarian_init(a, b):
    """
    Initialize permutation using spectral embedding for locations and 
    Hungarian algorithm for optimal assignment based on surrogate costs.
    """
    n = len(a)
    
    # Spectral embedding for locations: use top eigenvectors of B
    B_sym = (b + b.T) / 2.0
    try:
        evals, evecs = eigh(B_sym)
        # Use the top few eigenvectors for a richer embedding
        # Take the last 3 eigenvectors (largest eigenvalues) for 3D embedding
        k = min(3, n)
        V = evecs[:, -k:]
        # Project locations onto this subspace
        loc_coords = V.T  # shape (k, n)
        
        # For facilities, use row sums of A as a proxy for "flow centrality"
        # and also use spectral embedding of A
        A_sym = (a + a.T) / 2.0
        evals_a, evecs_a = eigh(A_sym)
        V_a = evecs_a[:, -k:]
        fac_coords = V_a.T  # shape (k, n)
        
        # Compute surrogate cost matrix: for each facility i and location j,
        # use the dot product of their embeddings as a similarity measure
        # We want to minimize flow * distance, so high flow should go to low distance
        # Use negative dot product as a cost proxy
        surrogate = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                surrogate[i, j] = -np.dot(fac_coords[:, i], loc_coords[:, j])
        
        # Solve Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
        
    except Exception:
        # Fallback to simple sorting
        loc_order = np.argsort(b.sum(axis=1))
        fac_order = np.argsort(a.sum(axis=1))
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

def _best_improvement_ls(a, b, p, time_limit, start_time):
    """
    Best-improvement local search: repeatedly find the best swap and apply it.
    Stops when no improving swap is found or time limit is reached.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Vectorized search for best swap
        # For efficiency, we can sample or use full search for n=100
        # n^2 = 10000 swaps, each O(n) = 100 ops, so ~1M ops per pass
        # This is fast enough for a few passes
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # No improving swap found
        
        # Apply the best swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
    
    return p, current_cost

def _perturb(p, n, strength=5):
    """
    Perturb the permutation by performing 'strength' random swaps.
    """
    p = p.copy()
    for _ in range(strength):
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s kill
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions using spectral-Hungarian init
    # Run multiple independent local searches from different starting points
    
    # First, get a high-quality initial solution
    p_init = _spectral_hungarian_init(a, b)
    cost_init = _qap_cost(a, b, p_init)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Multi-Start ILS
    # We'll run multiple short ILS cycles, each starting from a perturbation
    # of the current best solution
    
    num_starts = 10  # Number of independent ILS runs
    time_per_start = (deadline - 10) / num_starts  # Reserve 10s for final polish
    
    for s in range(num_starts):
        if time.time() - start > deadline - 5:
            break
        
        # Start from a perturbation of the best solution found so far
        p_cur = _perturb(best_p, n, strength=3 + s % 5)
        cost_cur = _qap_cost(a, b, p_cur)
        
        # Run local search to convergence
        time_limit_for_ls = time.time() - start + time_per_start
        p_ls, cost_ls = _best_improvement_ls(a, b, p_cur, time_limit_for_ls, start)
        
        # Update best if improved
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 3: Final polish with a more aggressive local search
    if time.time() - start < deadline - 2:
        time_remaining = deadline - (time.time() - start) - 1
        p_final, cost_final = _best_improvement_ls(a, b, best_p.copy(), 
                                                    time.time() - start + time_remaining, start)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    return best_p.tolist()
