import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Try to find a rank-32 decomposition using minimum weight perfect matching
    # Strategy: For each output element C[k,i], we need to compute sum_j A[i,j]*B[j,k]
    # We can use a tensor decomposition where each rank-1 term contributes to multiple outputs
    
    # Let's try a structured approach: use 32 rank-1 terms
    # Each term r has U_r, V_r, W_r vectors
    # U_r is in R^(n*m) = R^8, V_r is in R^(m*p) = R^32, W_r is in R^(p*n) = R^16
    
    # The decomposition must satisfy:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = delta_ij * delta_jk (the multiplication tensor)
    
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = delta_ij * delta_jk
    # where i is row of A, j is column of A / row of B, k is column of B
    
    # Let's use a greedy approach to find a low-rank decomposition
    # Start with the standard decomposition and try to merge terms
    
    # Standard decomposition: for each (i,j,k) where j is shared, we have a term
    # That gives n*m*p = 2*4*8 = 64 terms
    
    # But we can do better. Let's use the fact that we only need n*p = 16 output elements
    # and m = 4 intermediate dimensions
    
    # Try rank 32: use 32 terms
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We'll use a structured approach based on splitting the computation
    # For each output C[k,i] = sum_j A[i,j] * B[j,k]
    # We can group the j's and use clever combinations
    
    # Let's try a direct construction with 32 terms
    # Each term will handle a subset of the (i,j,k) triples
    
    # Term r: let r = 4*a + b where a in {0,1,2,3} and b in {0,1,2,3}
    # This gives us 16 terms, but we need 32, so let's use a different structure
    
    # Actually, let's try to use the known optimal bound for 2x4x8 matrices
    # The theoretical minimum is at least n*p = 16, but likely higher
    
    # Let's try a systematic search for rank 32
    # We'll set up the equations and solve them
    
    # For each (i,j,k), the constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j matches, else 0
    
    # This is a nonlinear system. Let's try a different approach.
    
    # Use a Kronecker product structure: split into 4 blocks of 2x2x2
    # Each 2x2x2 can be done in rank 7 (Strassen), so 4*7 = 28
    
    # But we need to handle the rectangular structure properly
    
    # Let's use 4 Strassen blocks, each handling a 2x2x2 subproblem
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into 2 blocks of 2x2, B into 4 blocks of 2x2, C into 4 blocks of 2x2
    
    # Block (i_block, j_block, k_block) where i_block in {0}, j_block in {0,1}, k_block in {0,1,2,3}
    # That's 1*2*4 = 8 blocks, each 2x2x2
    
    # With Strassen, each block needs rank 7, so total rank = 8*7 = 56
    
    # To get rank 32, we need to share computations between blocks
    
    # Let's try a different decomposition: use 32 terms where each term handles multiple blocks
    
    # Term r: assign it to handle a specific (i, j_pair, k_pair) combination
    # where j_pair is a pair of j's and k_pair is a pair of k's
    
    # For i in {0,1}, j_pair in {(0,1), (2,3)}, k_pair in {(0,1), (2,3), (4,5), (6,7)}
    # That's 2 * 2 * 4 = 16 combinations, but we need 32 terms
    
    # Let's use 2 terms per combination, one for each i value
    # Actually, let's just try rank 40 and see if it works
    
    # For now, return the rank 56 solution as a fallback
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 decomposition patterns
    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    blocks = []
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            blocks.append((j_off, k_off))
    
    for idx, (j_off, k_off) in enumerate(blocks):
        base_rank = idx * 7
        add_block(base_rank, 0, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
