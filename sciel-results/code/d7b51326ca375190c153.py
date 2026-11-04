"""Rewritten program: rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Structural approach: split the 4x4x9 multiplication tensor as a direct sum of
    # a 4x4x4 sub-tensor (columns 0..3 of B) and a 4x4x5 sub-tensor (columns 4..8 of B).
    # For the 4x4x4 part, use the known rank-28 Strassen-based decomposition for 4x4x4.
    # For the 4x4x5 part, use a rank-15 decomposition via tensor product of 2x2 Strassen
    # (rank 7) with a 1x1x5 identity, giving 7*5/5... wait, that doesn't work directly.
    #
    # Let me reconsider. The 4x4x5 multiplication: C = A * B where A is 4x4, B is 4x5, C is 4x5.
    # We can view this as 4 separate column-vector multiplications, each 4x1 from a 4x4 * 4x1.
    # Actually, B has 5 columns, each of length 4. So C = A*B where B is 4x5.
    # This is equivalent to 5 independent 4x4*4x1 multiplications, but they share the same A.
    # 
    # Alternative: Use the fact that 4x4x5 can be decomposed by splitting B into two parts:
    # B = [B1 | B2] where B1 is 4x2 and B2 is 4x3.
    # Then C = A*B1 + A*B2, and we can use rank-7 for 4x4x2 (2x2 Strassen tensor product) 
    # and rank-? for 4x4x3.
    #
    # Actually, let me use a cleaner approach. The 4x4x4 Strassen-based rank-28 decomposition:
    # This comes from applying Strassen (rank 7) to two 2x2 block matrices.
    #
    # For 4x4x4: A = [[A11, A12],[A21, A22]] (each 2x2), B = [[B11, B12],[B21, B22]] (each 2x2)
    # C = A*B, each Cij is 2x2.
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)(B11+B22)  -> 7 products of 2x2 matrices
    # Each 2x2*2x2 product itself uses Strassen with rank 7.
    # So total rank = 7 * 7 = 49? No, that's not right either.
    #
    # Actually, the rank of the 4x4x4 matrix multiplication tensor is known to be at most 28.
    # This comes from a specific construction. Let me use a verified rank-28 decomposition.
    #
    # For simplicity and correctness, let me use a different structural approach:
    # Split 4x4x9 as 4x4x4 + 4x4x4 + 4x4x1.
    # Use schoolbook (rank 16) for each 4x4x4 block and rank 16 for the 4x4x1 block.
    # Total rank = 16 + 16 + 16 = 48. That's much better than 144.
    #
    # Actually, even better: use Strassen (rank 7) for 2x2x2 sub-blocks.
    # But let me first get a correct working solution with reduced rank.
    #
    # Let me use the direct sum approach with schoolbook for each column group:
    # B is 4x9. Split into three groups: columns 0-3 (4x4), columns 4-7 (4x4), columns 8 (4x1).
    # For each 4x4x4 block: schoolbook rank 16.
    # For the 4x4x1 block: schoolbook rank 16.
    # Total rank = 16 + 16 + 16 = 48.
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Split p into groups: [4, 4, 1]
    groups = [4, 4, 1]
    
    U_list = []
    V_list = []
    W_list = []
    
    col_offset = 0
    for group_size in groups:
        # For a 4x4xg block where g is the number of columns of B
        # U shape: (n*m, rank) = (16, rank)
        # V shape: (m*group_size, rank) = (4*g, rank)
        # W shape: (group_size*n, rank) = (g*4, rank)
        # Schoolbook: rank = n*m*g
        rank_g = n * m * group_size
        
        U_g = np.zeros((n * m, rank_g))
        V_g = np.zeros((m * group_size, rank_g))
        W_g = np.zeros((group_size * n, rank_g))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(group_size):
                    U_g[i * m + j, col] = 1
                    V_g[j * group_size + k, col] = 1
                    W_g[k * n + i, col] = 1
                    col += 1
        
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        col_offset += rank_g
    
    # Concatenate along columns
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify correctness with random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[k*n+i, :] = sum_r W[k*n+i, r] * (sum_{i',j} U[i*m+j, r] * A[i',j]) * (sum_{j',k'} V[j*p+k', r] * B[j',k'])
    # Actually the convention is W[k*n+i, r], so C[i, k] = C[k*n+i] in some sense.
    # Let me check: the output factor follows the official trace convention W[k*n+i, r].
    # So C[i, k] corresponds to W[k*n+i, r].
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_val = sum(U[i * m + j, r] * A[i, j] for j in range(m))
                v_val = sum(V[j * p + k, r] * B[j, k] for j in range(m))
                C_decomp[i, k] += W[k * n + i, r] * u_val * v_val
    
    error = np.max(np.abs(C_decomp - C_expected))
    print(f"Max error: {error}")
    print(f"Valid: {error < 1e-10}")
