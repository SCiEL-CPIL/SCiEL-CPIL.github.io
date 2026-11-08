"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_block_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    return error < 1e-6

def _find_rank_10_block():
    """
    Attempt to find a rank 10 decomposition for M_{2,3,2}.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    n, m, p = 2, 3, 2
    target_rank = 10
    time_limit = 20  # Increased time limit
    
    start_time = time.time()
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
        
        # Check
        if _verify_block_decomposition(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_blocks = 5
    block_size = 2
    
    # Try to find a rank 10 block first
    block_decomp = _find_rank_10_block()
    
    if block_decomp is not None:
        rank_per_block = 10
        total_rank = num_blocks * rank_per_block
        U_block, V_block, W_block = block_decomp
        
        U_base = np.zeros((n * m, total_rank))
        V_base = np.zeros((m * p, total_rank))
        W_base = np.zeros((p * n, total_rank))
        
        for b in range(num_blocks):
            col_offset = b * rank_per_block
            U_base[:, col_offset:col_offset+rank_per_block] = U_block
            for j in range(m):
                for k_local in range(block_size):
                    k_global = b * block_size + k_local
                    V_base[j * p + k_global, col_offset:col_offset+rank_per_block] = V_block[j * block_size + k_local, :]
            for k_local in range(block_size):
                for i in range(n):
                    k_global = b * block_size + k_local
                    W_base[k_global * n + i, col_offset:col_offset+rank_per_block] = W_block[k_local * n + i, :]
        
        best_U, best_V, best_W = U_base, V_base, W_base
    else:
        # Fallback to rank 11 block
        rank_per_block = 11
        total_rank = num_blocks * rank_per_block
        
        U_base = np.zeros((n * m, total_rank))
        V_base = np.zeros((m * p, total_rank))
        W_base = np.zeros((p * n, total_rank))
        
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
    
    # Convert to lists for compatibility
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
