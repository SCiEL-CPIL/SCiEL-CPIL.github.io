"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    We want to swap p[i] and p[j].
    
    Let x = p[i], y = p[j].
    The cost is sum_{u,v} a[u,v] * b[p[u], p[v]].
    
    After swapping p[i] and p[j]:
    - For the pair (i,j): a[i,j]*b[y,x] + a[j,i]*b[x,y] instead of a[i,j]*b[x,y] + a[j,i]*b[y,x]
    - For pairs (i,k) with k != i,j: a[i,k]*b[y,p[k]] instead of a[i,k]*b[x,p[k]]
    - For pairs (k,i) with k != i,j: a[k,i]*b[p[k],y] instead of a[k,i]*b[p[k],x]
    - For pairs (j,k) with k != i,j: a[j,k]*b[x,p[k]] instead of a[j,k]*b[y,p[k]]
    - For pairs (k,j) with k != i,j: a[k,j]*b[p[k],x] instead of a[k,j]*b[p[k],y]
    
    Delta = (a[i,j]*b[y,x] + a[j,i]*b[x,y] - a[i,j]*b[x,y] - a[j,i]*b[y,x])
           + sum_{k != i,j} [a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
                            + a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y])]
    """
    x = p[i]
    y = p[j]
    
    # Direct pair contribution
    delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
    
    # Interactions with all other facilities
    # Vectorized computation
    km = np.ones(len(p), dtype=bool)
    km[i] = False
    km[j] = False
    km_idx = np.where(km)[0]
    
    if len(km_idx) > 0:
        pk = p[km_idx]
        
        # a[i,k] terms
        ai_k = a[i, km_idx]
        delta += np.sum(ai_k * (b[y, pk] - b[x, pk]))
        
        # a[k,i] terms
        ak_i = a[km_idx, i]
        delta += np.sum(ak_i * (b[pk, y] - b[pk, x]))
        
        # a[j,k] terms
        aj_k = a[j, km_idx]
        delta += np.sum(aj_k * (b[x, pk] - b[y, pk]))
        
        # a[k,j] terms
        ak_j = a[km_idx, j]
        delta += np.sum(ak_j * (b[pk, x] - b[pk, y]))
    
    return int(delta)


def _greedy_init(a, b, n):
    """Greedy initialization based on row sums."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_init(n):
    """Random permutation initialization."""
    return np.random.permutation(n)


def _tabu_search(a, b, p, time_limit, tabu_tenure=30, max_iters=None):
    """Tabu search with 1-swap neighborhood."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_list = {}
    iteration = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if max_iters is not None and iteration >= max_iters:
            break
        
        iteration += 1
        
        # Best improving 1-swap
        best_swap_i, best_swap_j = -1, -1
        best_delta = 0
        
        # Sample a subset of swaps for speed
        sample_size = min(1500, n * n // 4)
        
        if sample_size < n * n:
            indices = np.random.choice(n * n, sample_size, replace=False)
            i_arr = indices // n
            j_arr = indices % n
            pairs = [(i_arr[k], j_arr[k]) for k in range(sample_size) if i_arr[k] != j_arr[k]]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for i, j in pairs:
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in tabu_list and tabu_list[key] > iteration:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            if delta < best_delta:
                best_delta = delta
                best_swap_i, best_swap_j = i, j
        
        if best_swap_i == -1:
            break
        
        # Perform the swap
        i, j = best_swap_i, best_swap_j
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iteration + tabu_tenure
        
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Clean up expired tabu entries occasionally
        if iteration % 500 == 0:
            expired = [k for k, v in tabu_list.items() if v <= iteration]
            for k in expired:
                del tabu_list[k]
    
    return best_p, best_cost


def _fw_birkhoff(a, b, p_init, time_limit, tol=1e-4):
    """
    Frank-Wolfe on Birkhoff polytope for QAP.
    Uses Hungarian projection to get convex jumps in assignment space.
    """
    n = len(p_init)
    
    # Start from a doubly stochastic matrix derived from p_init
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # Precompute gradient components
    # Gradient of f(X) = sum_{i,j,k,l} a[i,j] * X[i,k] * X[j,l] * b[k,l]
    # wrt X[i,k] is: sum_{j,l} a[i,j] * X[j,l] * b[k,l] + sum_{j,l} a[j,i] * X[j,l] * b[l,k]
    # (using symmetry of a and b)
    
    best_X = X.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    start_time = time.time()
    
    for it in range(500):
        if time.time() - start_time >= time_limit:
            break
        
        # Compute gradient
        # G[i,k] = sum_j a[i,j] * (B X^T)[j,k] + sum_j a[j,i] * (B^T X)[j,k]
        # where B = b
        BX = b @ X  # (n, n)
        BXT = b.T @ X.T  # (n, n)
        
        G = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            G[i, :] = a[i, :] @ BX.T + a[:, i] @ BXT.T
        
        # Find vertex of Birkhoff polytope minimizing linear objective G
        # This is the Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(G)
        S = np.zeros((n, n), dtype=np.float64)
        S[row_ind, col_ind] = 1.0
        
        # Frank-Wolfe step: line search
        # X_new = (1 - gamma) * X + gamma * S
        # Minimize f((1-gamma)*X + gamma*S) over gamma in [0,1]
        
        # Simple: try a few gamma values
        best_gamma = 0.0
        best_obj = np.inf
        
        for gamma in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            X_trial = (1 - gamma) * X + gamma * S
            # Round to permutation
            row_ind_r, col_ind_r = linear_sum_assignment(-X_trial)
            p_trial = np.zeros(n, dtype=np.int64)
            p_trial[row_ind_r] = col_ind_r
            cost_trial = _qap_cost(a, b, p_trial)
            if cost_trial < best_obj:
                best_obj = cost_trial
                best_gamma = gamma
        
        if best_gamma == 0.0:
            break
        
        X = (1 - best_gamma) * X + best_gamma * S
        
        # Check convergence
        if it % 10 == 0:
            row_ind_c, col_ind_c = linear_sum_assignment(-X)
            p_check = np.zeros(n, dtype=np.int64)
            p_check[row_ind_c] = col_ind_c
            cost_check = _qap_cost(a, b, p_check)
            if cost_check < best_cost:
                best_cost = cost_check
                best_X = X.copy()
    
    # Final rounding
    row_ind_f, col_ind_f = linear_sum_assignment(-best_X)
    p_final = np.zeros(n, dtype=np.int64)
    p_final[row_ind_f] = col_ind_f
    
    return p_final, best_cost


def _solve_tai256c(a, b):
    n = len(a)
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    inits.append(_greedy_init(a, b, n))
    inits.append(_random_init(n))
    inits.append(_random_init(n))
    
    # Phase 2: Apply Frank-Wolfe on Birkhoff polytope to each init
    best_p = None
    best_cost = np.inf
    fw_time_per = 25.0
    
    for i, p_init in enumerate(inits):
        if time.time() >= deadline - 30:
            break
        
        p_fw, cost_fw = _fw_birkhoff(a, b, p_init, time_limit=fw_time_per)
        
        if cost_fw < best_cost:
            best_cost = cost_fw
            best_p = p_fw.copy()
    
    # Phase 3: Tabu polish from best FW solution
    if best_p is not None and time.time() < deadline - 5:
        polish_time = min(20.0, deadline - time.time() - 5)
        p_polished, cost_polished = _tabu_search(a, b, best_p, time_limit=polish_time)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # If no FW solution was found, fall back to tabu from greedy init
    if best_p is None:
        p_greedy = _greedy_init(a, b, n)
        best_p, best_cost = _tabu_search(a, b, p_greedy, time_limit=min(30.0, deadline - time.time() - 5))
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
