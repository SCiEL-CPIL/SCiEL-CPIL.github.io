"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]
DEADLINE = 120.0  # seconds, with safety margin

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 8

    # Try to find a low-rank decomposition
    # Strategy: split into two 2x3 x 3x4 subproblems
    # Find rank-11 decomposition for 2x3 x 3x4, total rank = 22
    
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]  # (6, rank)
    V = [[0.0] * rank for _ in range(m * p)]  # (24, rank)
    W = [[0.0] * rank for _ in range(p * n)]  # (16, rank)
    
    # For each of two 3x4 blocks of B (cols 0-3 and 4-7):
    for block in range(2):
        base = block * 11
        k_offset = block * 4  # column offset in B and C
        
        # Rank-11 decomposition for 2x3 * 3x4
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # for i in {0,1}, k in {0,1,2,3}
        
        # Use a structured decomposition:
        # For each output column k (0-3), we need to compute 2 values
        # That's 8 outputs total from 12 input terms
        
        # Approach: use rank-4 for 2x2*2x2 (Strassen-like) + rank-4 for 2x1*1x2 + rank-3 for cross terms
        # Actually, let's use a direct but optimized approach
        
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,3])  -- contributes to C[0,0], C[1,3]
        # M2: (A[1,0]+A[1,1]) * B[0,0]            -- contributes to C[1,0]
        # M3: A[0,0] * (B[1,0]-B[0,0])            -- contributes to C[0,0]
        # M4: (A[0,1]-A[1,1]) * (B[1,0]+B[1,3])   -- contributes to C[0,0], C[0,3]
        # M5: A[0,1] * (B[0,0]+B[0,3])            -- contributes to C[0,0], C[0,3]
        # M6: (A[1,0]-A[0,0]) * (B[0,3]-B[1,3])   -- contributes to C[1,3]
        # M7: (A[0,0]+A[0,1]) * B[1,3]            -- contributes to C[0,3], C[1,3]
        
        # That's 7 for the 2x2*2x2 part (using A cols 0,1 and B rows 0,1, cols 0,3)
        # Plus we need to handle:
        # - B row 2 with all 4 columns: 2*4 = 8 terms, but can share
        # - B cols 1,2 with A cols 0,1
        
        # This is getting complex. Let me try a different approach.
        
        # Actually, let me just use a verified rank-11 scheme:
        # Direct: 2*3*4 = 24 terms, but we can do better
        
        # Use: for each of 4 columns, 2x3*3x1 = 2x1
        # 2x3*3x1 has rank 3 (just 3 multiplications)
        # So 4 columns * 3 = 12, not 11
        
        # Let me try rank 12 per block, total 24
        # Or try to find if rank 11 is possible
        
        # For now, use rank 12 per block:
        # For each column k in 0..3:
        #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
        # Each column needs 3 rank-1 terms, total 4*3 = 12
        
        # But we can potentially share some structure. Let me just use 12 per block.
        
        # Actually, let me reconsider. The original was 44 = 4 * 11.
        # If I use 2 blocks of 3x4 with rank 12 each, that's 24, much better!
        
        # For block b, column k in 0..3:
        for k in range(4):
            col = k_offset + k
            for j in range(3):
                c = base + k * 3 + j
                U[0 * 3 + j][c] += 1  # A[0,j]
                U[1 * 3 + j][c] += 1  # A[1,j]
                V[j * 8 + col][c] += 1  # B[j, col]
                W[col * 2 + 0][c] += 1  # C[0, col]
                W[col * 2 + 1][c] += 1  # C[1, col]
    
    # Verify
    if time.time() - t0 > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    A_test = np.random.randn(2, 3)
    B_test = np.random.randn(3, 8)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((2, 8))
    for c in range(rank):
        U_mat = U_np[:, c].reshape(2, 3)
        V_mat = V_np[:, c].reshape(3, 8)
        W_vec = W_np[:, c]
        prod = U_mat @ V_mat
        for i in range(2):
            for k in range(8):
                C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
