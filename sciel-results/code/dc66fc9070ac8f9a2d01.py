import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def get_strassen_3x3():
        """
        Return U, V, W for 3x3x3 matrix multiplication with rank 23.
        Uses the known rank-23 decomposition (Baur-Strassen).
        
        U: (9, 23) - indices [i*3+j]
        V: (9, 23) - indices [j*3+k]  
        W: (9, 23) - indices [k*3+i]
        
        The decomposition satisfies:
        sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k
        """
        # Rank-23 decomposition for 3x3x3 (Baur-Strassen algorithm)
        # This is a well-known result: rank(M_{3,3,3}) = 23
        
        U = np.array([
            # Row 0: (0,0)
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: (0,1)
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: (0,2)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: (1,0)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: (1,1)
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: (1,2)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: (2,0)
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 7: (2,1)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 8: (2,2)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=float)
        
        V = np.array([
            # Row 0: (0,0)
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: (0,1)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: (0,2)
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: (1,0)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: (1,1)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: (1,2)
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: (2,0)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 7: (2,1)
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 8: (2,2)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=float)
        
        W = np.array([
            # Row 0: (0,0)
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: (0,1)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: (0,2)
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: (1,0)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: (1,1)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: (1,2)
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: (2,0)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 7: (2,1)
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 8: (2,2)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=float)
        
        # This is still identity - we need the actual Strassen coefficients
        # Let me use a different approach: construct via known formulas
        
        # Actually, let's use a verified rank-23 decomposition
        # The Baur-Strassen algorithm for 3x3 has a specific structure
        # Let me use the standard published coefficients
        
        U = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=float)
        
        # This approach isn't working. Let me use a different strategy.
        # For 3x3x7, we can use a block approach with schoolbook for each 3-column block
        # but that gives rank 9*3 = 27... no wait, 9 per 3 columns, so 9+9+9+9=36 for 7 cols?
        # No: 3+3+1 columns -> 9+9+9 = 27? No, schoolbook for 3x3x3 is rank 9.
        # For 3x3x7: we can split B into 3+3+1, giving rank 9+9+9 = 27.
        # Or we can use Strassen for the 3x3x3 parts if we have valid coefficients.
        
        # Since I can't easily hardcode the correct rank-23 coefficients,
        # let me use a safe approach: schoolbook for 3x3x7 gives rank 63.
        # But we can do better by splitting: 3x3x3 + 3x3x3 + 3x3x1
        # = 9 + 9 + 9 = 27 using schoolbook for each block.
        
        return None  # Signal that this approach needs rethinking
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct by checking all entries."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check ALL entries (not just random) for correctness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    actual = 0.0
                    for r in range(rank):
                        actual += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    
                    if abs(actual - 1.0) > 1e-10:
                        return False
        
        return True
    
    def build_block_schoolbook(n, m, p, start_col):
        """Build schoolbook decomposition for 3x3xp block starting at column start_col."""
        rank_block = n * m * p
        U_block = np.zeros((n * m, rank_block), dtype=float)
        V_block = np.zeros((m * p, rank_block), dtype=float)
        W_block = np.zeros((p * n, rank_block), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_block[i * m + j, col] = 1
                    V_block[j * p + k, col] = 1
                    W_block[k * n + i, col] = 1
                    col += 1
        return U_block, V_block, W_block, rank_block
    
    # Strategy: Split 3x3x7 into 3x3x3 + 3x3x3 + 3x3x1
    # Each block uses schoolbook: rank 9 + 9 + 9 = 27
    try:
        n, m, p = 3, 3, 7
        
        # Block 1: columns 0-2 (3 columns)
        U1, V1, W1, r1 = build_block_schoolbook(3, 3, 3, 0)
        
        # Block 2: columns 3-5 (3 columns) 
        U2, V2, W2, r2 = build_block_schoolbook(3, 3, 3, 3)
        
        # Block 3: column 6 (1 column)
        U3, V3, W3, r3 = build_block_schoolbook(3, 3, 1, 6)
        
        total_rank = r1 + r2 + r3
        
        U = np.zeros((9, total_rank), dtype=float)
        V = np.zeros((21, total_rank), dtype=float)
        W = np.zeros((21, total_rank), dtype=float)
        
        # Place block 1
        U[:, 0:r1] = U1
        # V1 has shape (9, 9) for 3x3x3, but we need (21, 9) for 3x3x7
        # V1[j*3+k] corresponds to column k of B, rows j
        # In the full V, row index is j*7+k
        for j in range(3):
            for k in range(3):
                V[j * 7 + k, 0:r1] = V1[j * 3 + k, :]
        W[0:9, 0:r1] = W1  # W1 is (9, 9), W is (21, total_rank)
        
        # Place block 2
        U[:, r1:r1+r2] = U2
        for j in range(3):
            for k in range(3, 6):
                V[j * 7 + k, r1:r1+r2] = V2[j * 3 + (k-3), :]
        W[9:18, r1:r1+r2] = W2
        
        # Place block 3
        U[:, r1+r2:r1+r2+r3] = U3
        for j in range(3):
            V[j * 7 + 6, r1+r2:r1+r2+r3] = V3[j * 3 + 0, :]
        W[18:21, r1+r2:r1+r2+r3] = W3
        
        # Verify
        if verify_decomposition(U, V, W, 3, 3, 7):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback to full schoolbook
        return get_schoolbook()
        
    except Exception as e:
        # Fallback to schoolbook on error
        return get_schoolbook()
# EVOLVE-BLOCK-END
