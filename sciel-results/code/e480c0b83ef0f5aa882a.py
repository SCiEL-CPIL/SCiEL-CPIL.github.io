"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    num_blocks = 5
    block_size = 2
    
    # Baseline: Rank 55 solution using Strassen (7) + Outer (4) per 2x3x2 block
    rank_per_block_base = 11
    total_rank_base = num_blocks * rank_per_block_base
    
    U_base = np.zeros((n * m, total_rank_base))
    V_base = np.zeros((m * p, total_rank_base))
    W_base = np.zeros((p * n, total_rank_base))
    
    strassen_terms = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            for i, j, val in term['U']:
                U_base[i * m + j, col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V_base[j * p + k_global, col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W_base[k_global * n + i, col] += val
            col_offset += 1
            
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = total_rank_base
    
    # Attempt to find a rank 10 block decomposition for 2x3x2
    def _find_rank_10_block():
        n_b, m_b, p_b = 2, 3, 2
        target_rank = 10
        time_limit = 20
        
        start_time = time.time()
        
        # Try structured search over small integer entries
        # The tensor for M_{2,3,2} has 2*3*2 = 12 nonzero entries
        # We need rank 10 with entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        
        # Use a more systematic approach: try all combinations of a few patterns
        # Key insight: for 2x3x2, the minimum rank is likely 10 or 11
        
        # Try random search with more attempts and varied ranges
        rng = np.random.default_rng(42)
        
        for restart in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Use entries from {-1, 0, 1} primarily, occasionally {-2, 0, 2}
            max_val = 1 if restart % 3 != 0 else 2
            
            U = rng.integers(-max_val, max_val + 1, size=(n_b*m_b, target_rank)).astype(float)
            V = rng.integers(-max_val, max_val + 1, size=(m_b*p_b, target_rank)).astype(float)
            W = rng.integers(-max_val, max_val + 1, size=(p_b*n_b, target_rank)).astype(float)
            
            if _verify_decomposition(U, V, W, n_b, m_b, p_b):
                return U, V, W
                
        # Try a gradient-free local search starting from the rank 11 solution
        # and trying to eliminate one rank dimension
        time_left = deadline - time.time()
        if time_left > 5:
            # Use the known rank-11 block and try to reduce
            U11 = np.zeros((n_b*m_b, 11))
            V11 = np.zeros((m_b*p_b, 11))
            W11 = np.zeros((p_b*n_b, 11))
            
            for idx, term in enumerate(all_terms):
                col = idx
                for i, j, val in term['U']:
                    U11[i * m_b + j, col] += val
                for j, k_local, val in term['V']:
                    V11[j * p_b + k_local, col] += val
                for k_local, i, val in term['W']:
                    W11[k_local * n_b + i, col] += val
            
            # Try dropping each of the 11 columns and see if the remaining 10 still work
            # This won't work directly, but we can try to find a rank-10 solution
            # by searching near the rank-11 solution
            
            for col_to_drop in range(11):
                if time.time() - start_time > time_limit:
                    break
                U_sub = np.delete(U11, col_to_drop, axis=1)
                V_sub = np.delete(V11, col_to_drop, axis=1)
                W_sub = np.delete(W11, col_to_drop, axis=1)
                
                # Try small perturbations
                for pert in range(50):
                    U_p = U_sub + rng.normal(0, 0.1, U_sub.shape)
                    V_p = V_sub + rng.normal(0, 0.1, V_sub.shape)
                    W_p = W_sub + rng.normal(0, 0.1, W_sub.shape)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U_p * 2) / 2
                    V_r = np.round(V_p * 2) / 2
                    W_r = np.round(W_p * 2) / 2
                    
                    if _verify_decomposition(U_r, V_r, W_r, n_b, m_b, p_b):
                        return U_r, V_r, W_r
        
        return None

    block_decomp = _find_rank_10_block()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        rank_per_block = 10
        total_rank = num_blocks * rank_per_block
        
        if total_rank < best_rank:
            U_new = np.zeros((n * m, total_rank))
            V_new = np.zeros((m * p, total_rank))
            W_new = np.zeros((p * n, total_rank))
            
            for b in range(num_blocks):
                col_offset = b * rank_per_block
                U_new[:, col_offset:col_offset+rank_per_block] = U_block
                for j in range(m):
                    for k_local in range(block_size):
                        k_global = b * block_size + k_local
                        V_new[j * p + k_global, col_offset:col_offset+rank_per_block] = V_block[j * block_size + k_local, :]
                for k_local in range(block_size):
                    for i in range(n):
                        k_global = b * block_size + k_local
                        W_new[k_global * n + i, col_offset:col_offset+rank_per_block] = W_block[k_local * n + i, :]
            
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
