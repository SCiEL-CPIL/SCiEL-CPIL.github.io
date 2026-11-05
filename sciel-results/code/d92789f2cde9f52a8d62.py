"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: n*m x rank, V: m*p x rank, W: p*n x rank
    
    # Start with schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find better decompositions using random restarts with pruning
    # Strategy: Try ranks from lower to upper, use random initialization with sparsity
    
    # Known lower bound considerations:
    # For 3x4x7, schoolbook is 84
    # Try various rank values
    
    min_possible_rank = n * p  # 21 (lower bound)
    max_rank_to_try = 70
    
    # Try ranks in increasing order, but start from a reasonable range
    rank_candidates = list(range(60, 85)) + list(range(50, 60)) + list(range(40, 50))
    rank_candidates = sorted(set(rank_candidates))
    
    for target_rank in rank_candidates:
        if time.time() > deadline - 10.0:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try multiple random restarts for this rank
        n_restarts = 5
        success = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with random sparse matrices
            # Use a structured approach: start with schoolbook and try to merge columns
            
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Apply sparsity: keep only a few non-zero entries per row
            density = min(1.0, 3.0 / target_rank)
            for row in range(U_rand.shape[0]):
                nnz = max(1, int(np.random.binomial(target_rank, density)))
                keep = np.random.choice(target_rank, nnz, replace=False)
                U_rand[row, :] = 0
                U_rand[row, keep] = 1
            for row in range(V_rand.shape[0]):
                nnz = max(1, int(np.random.binomial(target_rank, density)))
                keep = np.random.choice(target_rank, nnz, replace=False)
                V_rand[row, :] = 0
                V_rand[row, keep] = 1
            for row in range(W_rand.shape[0]):
                nnz = max(1, int(np.random.binomial(target_rank, density)))
                keep = np.random.choice(target_rank, nnz, replace=False)
                W_rand[row, :] = 0
                W_rand[row, keep] = 1
            
            # Check if this decomposition works
            # T_{ijk} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
            # We need T_{ijk} = 1 for all valid (i,j,k)
            
            # Efficient check: for each column c, the outer product structure
            # Let's verify by constructing the tensor and comparing
            
            # Build tensor from decomposition
            T_decomp = np.zeros((n, m, p))
            for c in range(target_rank):
                uc = U_rand[:, c]
                vc = V_rand[:, c]
                wc = W_rand[:, c]
                
                # uc has entries at positions i*m+j
                # vc has entries at positions j*p+k
                # wc has entries at positions k*n+i
                
                # For non-zero entries, add the outer product
                u_idx = np.nonzero(uc)[0]
                v_idx = np.nonzero(vc)[0]
                w_idx = np.nonzero(wc)[0]
                
                if len(u_idx) == 0 or len(v_idx) == 0 or len(w_idx) == 0:
                    continue
                
                for ui in u_idx:
                    u_val = uc[ui]
                    i_idx = ui // m
                    j_from_u = ui % m
                    
                    for vi in v_idx:
                        v_val = vc[vi]
                        j_from_v = vi // p
                        k_idx = vi % p
                        
                        if j_from_u != j_from_v:
                            continue
                        
                        for wi in w_idx:
                            w_val = wc[wi]
                            k_from_w = wi // n
                            i_from_w = wi % n
                            
                            if k_idx != k_from_w or i_idx != i_from_w:
                                continue
                            
                            T_decomp[i_idx, j_from_u, k_idx] += u_val * v_val * w_val
            
            # Check if T_decomp equals the identity tensor (all 1s)
            if np.allclose(T_decomp, 1.0, atol=1e-9):
                success = True
                best_rank = target_rank
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                break
        
        if success:
            # Try to go lower
            continue
    
    # Convert to lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
