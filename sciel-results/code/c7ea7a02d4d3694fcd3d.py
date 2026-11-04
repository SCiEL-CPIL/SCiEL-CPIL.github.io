"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Vectorized Best-Improvement 2-Opt Local Search.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    A_sym = a + a.T
    A_pos = A_sym[p].copy()
    diag_A_sym = np.diag(A_sym)
    
    while time.time() - start_time < time_limit:
        # Compute M = A_pos @ D.T
        # M[i, j] = A_pos[i] dot D[j]
        M = A_pos @ D.T
        diag_M = np.diag(M)
        diag_D = np.diag(D)
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        # To speed up, we can just scan i and vectorize j.
        for i in range(n):
            # Vectorized delta for fixed i
            # Delta(i, j) = (A_i - A_j) dot (D_j - D_i) + C_ij
            # = A_i dot D_j - A_i dot D_i - A_j dot D_j + A_j dot D_i + C_ij
            
            # A_i dot D_j for all j: M[i, :]
            # A_i dot D_i: M[i, i]
            # A_j dot D_j for all j: diag_M
            # A_j dot D_i for all j: M[:, i] (since M[j, i] = A_j dot D_i)
            
            vec_delta = M[i, :] - M[i, i] - diag_M + M[:, i]
            
            # Add corrections C[i, j]
            # C[i, j] = (A_sym[i, i] - A_sym[j, j]) * (D[j, j] - D[i, i]) + (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            
            # Term 1: (A_sym[i, i] - A_sym[j, j]) * (D[j, j] - D[i, i])
            term1 = (diag_A_sym[i] - diag_A_sym) * (diag_D - diag_D[i])
            
            # Term 2: (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            term2 = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            vec_delta += term1 + term2
            
            # Find best j
            min_val = np.min(vec_delta)
            if min_val < best_delta:
                best_delta = min_val
                best_j = np.argmin(vec_delta)
                best_i = i
        
        if best_i != -1 and best_delta < 0:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            
            # Update D
            D[i], D[j] = D[j].copy(), D[i].copy()
            D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
            
            # Update A_pos
            A_pos[i], A_pos[j] = A_pos[j].copy(), A_pos[i].copy()
            
            current_cost += best_delta
        else:
            break
            
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _eigen_init(a, b, n):
    """
    Initialize permutation using eigenvector-based ranking.
    This exploits the structure of the QAP by aligning the principal
    directions of the flow and distance matrices.
    """
    # Symmetrize matrices for eigen decomposition
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Get eigenvectors
    # We want the top eigenvector to capture the main structure
    try:
        _, vecs_A = np.linalg.eigh(A_sym)
        _, vecs_B = np.linalg.eigh(B_sym)
    except:
        return np.random.permutation(n)
    
    # Take the last eigenvector (largest eigenvalue)
    v_A = vecs_A[:, -1]
    v_B = vecs_B[:, -1]
    
    # Rank facilities by their projection on v_A
    # Rank locations by their projection on v_B
    rank_A = np.argsort(v_A)
    rank_B = np.argsort(v_B)
    
    # We want to align high values with high values (or low with low)
    # Let's try matching ranks
    # p[i] is the location assigned to facility i
    # So if facility i has rank r in A, it should go to location with rank r in B
    
    # Create inverse rank for B
    inv_rank_B = np.empty(n, dtype=int)
    inv_rank_B[rank_B] = np.arange(n)
    
    # Assign facility i to location with the same rank as i in A
    # rank_A gives the order of facilities. So facility at index rank_A[k] has rank k.
    # We want facility rank_A[k] to go to location rank_B[k].
    p = np.empty(n, dtype=int)
    for k in range(n):
        p[rank_A[k]] = rank_B[k]
        
    return p

def _rank_match_init(a, b, n):
    """
    Initialize permutation using row-sum rank matching.
    """
    sum_A = a.sum(axis=1)
    sum_B = b.sum(axis=1)
    
    rank_A = np.argsort(sum_A)
    rank_B = np.argsort(sum_B)
    
    p = np.empty(n, dtype=int)
    for k in range(n):
        p[rank_A[k]] = rank_B[k]
        
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # Safety margin for 150s limit
    
    # Try multiple initialization strategies and keep the best
    inits = []
    
    # 1. Random
    inits.append(np.random.permutation(n))
    
    # 2. Rank matching
    try:
        inits.append(_rank_match_init(a, b, n))
    except:
        pass
        
    # 3. Eigen-based
    try:
        inits.append(_eigen_init(a, b, n))
    except:
        pass
        
    # 4. Another random
    inits.append(np.random.permutation(n))
    
    best_p = None
    best_cost = np.inf
    
    # Evaluate initial solutions
    for p in inits:
        cost = _compute_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    # ILS loop
    current_p = best_p.copy()
    
    while time.time() - start_time < time_budget:
        # Local search
        p_ls, cost_ls = _local_search_vectorized(a, b, current_p, time_budget, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Check time
        if time.time() - start_time > time_budget - 5:
            break
            
        # Perturb
        # Adaptive perturbation: more if stuck, less if improving
        # Simple strategy: random k
        k = random.randint(3, 15)
        current_p = _perturb(p_ls, k)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
