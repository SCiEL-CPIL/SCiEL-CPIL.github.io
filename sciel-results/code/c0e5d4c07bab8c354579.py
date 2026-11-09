"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # We use a block decomposition.
    # Split p=10 into 5 blocks of size 2.
    # Each block is M_{2,3,2}.
    # Rank of M_{2,3,2} is 11.
    # Total rank = 5 * 11 = 55.
    
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Indices for A: i in {0,1}, j in {0,1,2}
    # Indices for B: j in {0,1,2}, k in {0,1} (local k)
    # Indices for C: i in {0,1}, k in {0,1} (local k)
    # W index: k * n + i
    
    # Strassen part (7 terms) for A[:, 0:2] and B[0:2, :]
    # Local indices: i, j in {0,1}, k in {0,1}
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00, C11
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 +=, C11 -=
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)] # C01 +=, C11 +=
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 +=, C10 +=
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 -=, C01 +=
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 +=
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)] # C00 +=
        }
    ]
    
    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    # Local indices: i in {0,1}, j=2, k in {0,1}
    outer_terms = [
        # i=0, k=0
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        # i=0, k=1
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        # i=1, k=0
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        # i=1, k=1
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            if time.time() > deadline:
                break
            col = col_offset
            # Fill U
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            # Fill V
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            # Fill W
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1
        if time.time() > deadline:
            break
            
    best_rank = total_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    # This is a structural refinement: check if the current decomposition has redundant terms
    # that can be merged or eliminated via linear combinations.
    
    if time.time() < deadline - 30:
        # Convert to numpy for efficient linear algebra
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check if the decomposition is valid first
        def _verify(U_mat, V_mat, W_mat, n, m, p):
            rank = U_mat.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[u_idx, c] * V_mat[v_idx, c] * W_mat[w_idx, c]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        if _verify(U_arr, V_arr, W_arr, n, m, p):
            # Try to find a lower rank decomposition by projecting onto a subspace
            # This is a heuristic approach: try to find linear combinations of the
            # rank-1 terms that yield a valid decomposition with fewer terms.
            
            # For now, try a simple approach: check if any term is a linear combination
            # of others in a way that preserves the decomposition.
            
            # A more sophisticated approach would use tensor rank minimization,
            # but that's computationally expensive. Instead, try small perturbations
            # and check if they reduce the effective rank.
            
            # Try to see if we can remove terms by solving a linear system
            # This is a placeholder for a more advanced optimization step.
            
            # If we have time, try a local search for rank reduction
            if time.time() < deadline - 10:
                # Try to find a null space vector that can be used to reduce rank
                # This is a heuristic and may not always work.
                
                # For each pair of terms, check if their difference is in the span
                # of the remaining terms. If so, we might be able to eliminate one.
                
                # This is a complex optimization problem, so for now, we'll just
                # verify the current decomposition and return it.
                
                pass
    
    # Convert back to lists if needed
    U_out = [[int(U[i][j]) if abs(U[i][j] - round(U[i][j])) < 1e-9 else U[i][j] for j in range(best_rank)] for i in range(n*m)]
    V_out = [[int(V[i][j]) if abs(V[i][j] - round(V[i][j])) < 1e-9 else V[i][j] for j in range(best_rank)] for i in range(m*p)]
    W_out = [[int(W[i][j]) if abs(W[i][j] - round(W[i][j])) < 1e-9 else W[i][j] for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
