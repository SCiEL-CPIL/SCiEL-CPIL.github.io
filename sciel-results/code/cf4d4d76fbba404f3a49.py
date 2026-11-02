"""Program for 3x5x5 matrix multiplication tensor decomposition using randomized Strassen sub-blocks and aggressive local search."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_2x2():
    """
    Return the 7 rank terms for 2x2x2 matrix multiplication using Strassen.
    """
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _build_decomposition_random(n, m, p, seed):
    """
    Build a decomposition using randomized Strassen blocks.
    We randomize the signs of the Strassen terms to create different structures.
    """
    rng = random.Random(seed)
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    rank = 0
    
    strassen_terms = _strassen_2x2_2x2_2x2()
    
    # Partitions for 3x5x5
    # i: 0,1,2 -> [0,1], [2]
    # j: 0,1,2,3,4 -> [0,1], [2,3], [4]
    # k: 0,1,2,3,4 -> [0,1], [2,3], [4]
    
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1], [2, 3], [4]]
    k_parts = [[0, 1], [2, 3], [4]]
    
    for i_range in i_parts:
        for j_range in j_parts:
            for k_range in k_parts:
                ni = len(i_range)
                nj = len(j_range)
                nk = len(k_range)
                
                if ni == 2 and nj == 2 and nk == 2:
                    i0, i1 = i_range[0], i_range[1]
                    j0, j1 = j_range[0], j_range[1]
                    k0, k1 = k_range[0], k_range[1]
                    
                    # Randomize signs for each term to break symmetry
                    for t_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_terms):
                        # Random global sign flip for this term
                        sign = rng.choice([1, -1])
                        
                        u_vec = [x * sign for x in u_vec]
                        v_vec = [x * sign for x in v_vec]
                        # W is not flipped by the same sign to maintain product? 
                        # Actually, if we flip U and V, W must be flipped to keep product same?
                        # (U*sign) * (V*sign) * W = U*V*W*sign^2 = U*V*W.
                        # So we can flip U and V together, or V and W, or U and W.
                        # Let's just flip U and V.
                        # w_vec stays same.
                        
                        for li in range(2):
                            for lj in range(2):
                                u_val = u_vec[li * 2 + lj]
                                if u_val != 0:
                                    i_idx = i_range[li]
                                    j_idx = j_range[lj]
                                    U[i_idx * m + j_idx, rank] = u_val
                        
                        for lj in range(2):
                            for lk in range(2):
                                v_val = v_vec[lj * 2 + lk]
                                if v_val != 0:
                                    j_idx = j_range[lj]
                                    k_idx = k_range[lk]
                                    V[j_idx * p + k_idx, rank] = v_val
                        
                        for lk in range(2):
                            for li in range(2):
                                w_val = w_vec[lk * 2 + li]
                                if w_val != 0:
                                    k_idx = k_range[lk]
                                    i_idx = i_range[li]
                                    W[k_idx * n + i_idx, rank] = w_val
                        
                        rank += 1
                else:
                    for i_idx in i_range:
                        for j_idx in j_range:
                            for k_idx in k_range:
                                U[i_idx * m + j_idx, rank] = 1
                                V[j_idx * p + k_idx, rank] = 1
                                W[k_idx * n + i_idx, rank] = 1
                                rank += 1
    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return U, V, W, rank

def _verify_vectorized(U, V, W, n, m, p):
    """
    Fast vectorized verification.
    """
    rank = U.shape[1]
    if rank == 0:
        return False
        
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    for k in range(p):
        V_part = V_t[:, k, :]  # (m, rank)
        W_part = W_t[k, :, :]  # (n, rank)
        
        A = U_t * W_part[:, np.newaxis, :] # (n, m, rank)
        T_k = (A * V_part[np.newaxis, :, :]).sum(axis=2) # (n, m)
        
        if np.any(np.abs(T_k - 1.0) > 1e-9):
            return False
            
    return True

def _try_remove_column(U, V, W, n, m, p, col_idx):
    rank = U.shape[1]
    if rank <= 1:
        return None, None, None
    U_new = np.delete(U, col_idx, axis=1)
    V_new = np.delete(V, col_idx, axis=1)
    W_new = np.delete(W, col_idx, axis=1)
    if _verify_vectorized(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _greedy_removal(U, V, W, n, m, p, deadline):
    """
    Aggressive greedy removal with random restarts of the removal order.
    """
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = U.shape[1]
        
        # Shuffle column order to avoid bias
        order = list(range(rank))
        random.shuffle(order)
        
        for col_idx in order:
            if (time.time() - start_time) > deadline:
                break
            U_new, V_new, W_new = _try_remove_column(U, V, W, n, m, p, col_idx)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                improved = True
                break
                
    return U, V, W

def _greedy_pairwise_removal(U, V, W, n, m, p, deadline):
    """
    Greedy pairwise removal.
    """
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = U.shape[1]
        if rank < 2:
            break
            
        pairs = [(i, j) for i in range(rank) for j in range(i+1, rank)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if (time.time() - start_time) > deadline:
                break
            mask = np.ones(rank, dtype=bool)
            mask[i] = False
            mask[j] = False
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify_vectorized(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                improved = True
                break
                
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try multiple random seeds for the Strassen construction
    num_attempts = 5
    time_per_attempt = (deadline - start_time - 10.0) / num_attempts
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
            
        seed = attempt * 1000 + 42
        U, V, W, rank = _build_decomposition_random(n, m, p, seed)
        
        if not _verify_vectorized(U, V, W, n, m, p):
            continue
            
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Refine this specific decomposition
        # Phase 1: Single removal
        phase1_deadline = min(deadline - 5.0, start_time + time_per_attempt * 0.5)
        U, V, W = _greedy_removal(U, V, W, n, m, p, phase1_deadline)
        
        # Phase 2: Pairwise removal
        if time.time() < deadline - 5.0:
            phase2_deadline = min(deadline - 5.0, start_time + time_per_attempt * 0.8)
            U, V, W = _greedy_pairwise_removal(U, V, W, n, m, p, phase2_deadline)
            
        # Phase 3: Final single removal
        if time.time() < deadline - 5.0:
            U, V, W = _greedy_removal(U, V, W, n, m, p, deadline - 5.0)
            
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
    # Fallback to schoolbook if nothing better found or if verification failed
    if best_U is None or not _verify_vectorized(best_U, best_V, best_W, n, m, p):
        rank = n * m * p
        best_U = np.zeros((n * m, rank))
        best_V = np.zeros((m * p, rank))
        best_W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1
                    best_V[j * p + k, col] = 1
                    best_W[k * n + i, col] = 1
                    col += 1
        best_rank = rank

    # Convert to lists and clean up
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx]) < 1e-10:
                    row[idx] = 0.0
                elif abs(row[idx] - round(row[idx])) < 1e-10:
                    row[idx] = float(round(row[idx]))
                    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
