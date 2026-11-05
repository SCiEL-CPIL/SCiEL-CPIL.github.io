import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 48 decomposition using recursive Strassen.
    # A is 2x4, B is 4x7.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x7) and B2 (2x7).
    # C = A1 B1 + A2 B2.
    
    # Helper to get Strassen decomposition for 2x2x2 (rank 7)
    # Returns U, V, W parts for a single 2x2x2 block.
    # Indices for A: 0,1 (rows), 0,1 (cols)
    # Indices for B: 0,1 (rows), 0,1 (cols)
    # Indices for C: 0,1 (rows), 0,1 (cols)
    
    def strassen_2x2x2():
        # Returns list of (u, v, w) tuples where u, v, w are vectors of length 2, 2, 2 (flattened)
        # Actually, we need to map to the global indices.
        # u is in A space (size 2x2=4)
        # v is in B space (size 2x2=4)
        # w is in C space (size 2x2=4)
        
        # Strassen terms:
        # M1 = A00 * B00
        # M2 = A01 * B11
        # M3 = (A00+A01) * (B00-B11)
        # M4 = (A10-A00) * (B00+B10)
        # M5 = A11 * (B01+B11)
        # M6 = (A01-A11) * (B10+B11)
        # M7 = (A00+A11) * (B00-B10)
        
        # C00 = M1 + M3 + M7
        # C01 = M5 + M6
        # C10 = M4 + M7
        # C11 = M2 + M5
        
        terms = []
        # M1
        u = [1, 0, 0, 0]
        v = [1, 0, 0, 0]
        w = [1, 0, 0, 0]
        terms.append((u, v, w))
        
        # M2
        u = [0, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [0, 0, 0, 1]
        terms.append((u, v, w))
        
        # M3
        u = [1, 1, 0, 0]
        v = [1, 0, 0, -1]
        w = [1, 0, 0, 0]
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 1, 0]
        v = [1, 0, 1, 0]
        w = [0, 0, 1, 0]
        terms.append((u, v, w))
        
        # M5
        u = [0, 0, 0, 1]
        v = [0, 1, 0, 1]
        w = [0, 1, 0, 0]
        terms.append((u, v, w))
        
        # M6
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [0, 1, 0, 0]
        terms.append((u, v, w))
        
        # M7
        u = [1, 0, 0, 1]
        v = [1, 0, 0, -1]
        w = [0, 0, 1, 0]
        terms.append((u, v, w))
        
        return terms

    def standard_2x2x1():
        # Rank 3 decomposition for 2x2x1
        # A (2x2), B (2x1), C (2x1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # M1 = A00 B0
        # M2 = A01 B1
        # M3 = (A10 + A11) (B0 + B1) - M1 - M2? No.
        # Rank 3 terms:
        # 1. A00 B0 -> C0
        # 2. A01 B1 -> C0
        # 3. (A10 - A00) B0 + (A11 - A01) B1 -> C1? 
        # This M3 is not a single product.
        # Let's use Rank 4 to be safe? 
        # If I use Rank 4, total is 7*3 + 4 = 25. Total 50.
        # If I use Rank 3, total is 7*3 + 3 = 24. Total 48.
        # I need the correct Rank 3 terms.
        
        # Correct Rank 3 for 2x2x1:
        # M1 = A00 B0
        # M2 = A01 B1
        # M3 = (A10 + A11) (B0 + B1)
        # C0 = M1 + M2
        # C1 = M3 - M1 - M2 + A10 B1 + A11 B0? No.
        
        # Actually, for 2x2x1, the rank is 3.
        # Terms:
        # 1. u=[1,0,0,0], v=[1,0], w=[1,0]  (A00 B0 -> C0)
        # 2. u=[0,1,0,0], v=[0,1], w=[1,0]  (A01 B1 -> C0)
        # 3. u=[0,0,1,1], v=[1,1], w=[0,1]  ((A10+A11)(B0+B1) -> C1)
        # Check C1: (A10+A11)(B0+B1) = A10 B0 + A10 B1 + A11 B0 + A11 B1.
        # We need C1 = A10 B0 + A11 B1.
        # Difference: A10 B1 + A11 B0.
        # This is not zero.
        
        # So Rank 3 is not this simple.
        # I will use Rank 4 for the 2x2x1 part to ensure correctness.
        # Rank 4:
        # 1. A00 B0 -> C0
        # 2. A01 B1 -> C0
        # 3. A10 B0 -> C1
        # 4. A11 B1 -> C1
        
        terms = []
        # 1
        u = [1, 0, 0, 0]
        v = [1, 0]
        w = [1, 0]
        terms.append((u, v, w))
        
        # 2
        u = [0, 1, 0, 0]
        v = [0, 1]
        w = [1, 0]
        terms.append((u, v, w))
        
        # 3
        u = [0, 0, 1, 0]
        v = [1, 0]
        w = [0, 1]
        terms.append((u, v, w))
        
        # 4
        u = [0, 0, 0, 1]
        v = [0, 1]
        w = [0, 1]
        terms.append((u, v, w))
        
        return terms

    # Build the full decomposition
    # Total Rank: 2 * (3 * 7 + 4) = 2 * 25 = 50.
    # This is better than 56.
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Process A1 (rows 0,1 of A; cols 0,1 of A) and B1 (rows 0,1 of B; cols 0..6 of B)
    # A1 is at A[0:2, 0:2]
    # B1 is at B[0:2, 0:7]
    
    # Split B1 into three 2x2 and one 2x1
    # B1 cols 0,1 -> B1_1
    # B1 cols 2,3 -> B1_2
    # B1 cols 4,5 -> B1_3
    # B1 cols 6   -> B1_4
    
    for block_idx in range(3):
        # Strassen 2x2x2
        terms = strassen_2x2x2()
        for u, v, w in terms:
            # u is 4x1 (A1 indices)
            # v is 4x1 (B1_block indices)
            # w is 4x1 (C1_block indices)
            
            # Map u to global U
            # A1 is at A[0:2, 0:2]
            # u[0] -> A[0,0] -> U[0]
            # u[1] -> A[0,1] -> U[1]
            # u[2] -> A[1,0] -> U[2]
            # u[3] -> A[1,1] -> U[3]
            U[0:4, col] = u
            
            # Map v to global V
            # B1_block is at B[0:2, 2*block_idx : 2*block_idx+2]
            # v[0] -> B[0, 2*block_idx] -> V[0 * 7 + 2*block_idx]
            # v[1] -> B[1, 2*block_idx] -> V[1 * 7 + 2*block_idx]
            # v[2] -> B[0, 2*block_idx+1] -> V[0 * 7 + 2*block_idx+1]
            # v[3] -> B[1, 2*block_idx+1] -> V[1 * 7 + 2*block_idx+1]
            base_col = 2 * block_idx
            V[0 * 7 + base_col, col] = v[0]
            V[1 * 7 + base_col, col] = v[1]
            V[0 * 7 + base_col + 1, col] = v[2]
            V[1 * 7 + base_col + 1, col] = v[3]
            
            # Map w to global W
            # C1_block is at C[0:2, 2*block_idx : 2*block_idx+2]
            # W index is k * n + i
            # w[0] -> C[0, 2*block_idx] -> W[2*block_idx * 2 + 0]
            # w[1] -> C[0, 2*block_idx+1] -> W[(2*block_idx+1) * 2 + 0]
            # w[2] -> C[1, 2*block_idx] -> W[2*block_idx * 2 + 1]
            # w[3] -> C[1, 2*block_idx+1] -> W[(2*block_idx+1) * 2 + 1]
            W[2*block_idx * 2 + 0, col] = w[0]
            W[(2*block_idx+1) * 2 + 0, col] = w[1]
            W[2*block_idx * 2 + 1, col] = w[2]
            W[(2*block_idx+1) * 2 + 1, col] = w[3]
            
            col += 1
            
    # Last block: 2x2x1
    terms = standard_2x2x1()
    for u, v, w in terms:
        U[0:4, col] = u
        base_col = 6
        V[0 * 7 + base_col, col] = v[0]
        V[1 * 7 + base_col, col] = v[1]
        W[6 * 2 + 0, col] = w[0]
        W[6 * 2 + 1, col] = w[1]
        col += 1
        
    # Process A2 (rows 0,1 of A; cols 2,3 of A) and B2 (rows 2,3 of B; cols 0..6 of B)
    # A2 is at A[0:2, 2:4]
    # B2 is at B[2:4, 0:7]
    
    for block_idx in range(3):
        terms = strassen_2x2x2()
        for u, v, w in terms:
            # Map u to global U
            # A2 is at A[0:2, 2:4]
            # u[0] -> A[0,2] -> U[0 * 4 + 2] = U[2]
            # u[1] -> A[0,3] -> U[0 * 4 + 3] = U[3]
            # u[2] -> A[1,2] -> U[1 * 4 + 2] = U[6]
            # u[3] -> A[1,3] -> U[1 * 4 + 3] = U[7]
            U[2, col] = u[0]
            U[3, col] = u[1]
            U[6, col] = u[2]
            U[7, col] = u[3]
            
            # Map v to global V
            # B2_block is at B[2:4, 2*block_idx : 2*block_idx+2]
            # v[0] -> B[2, 2*block_idx] -> V[2 * 7 + 2*block_idx]
            # v[1] -> B[3, 2*block_idx] -> V[3 * 7 + 2*block_idx]
            # v[2] -> B[2, 2*block_idx+1] -> V[2 * 7 + 2*block_idx+1]
            # v[3] -> B[3, 2*block_idx+1] -> V[3 * 7 + 2*block_idx+1]
            base_col = 2 * block_idx
            V[2 * 7 + base_col, col] = v[0]
            V[3 * 7 + base_col, col] = v[1]
            V[2 * 7 + base_col + 1, col] = v[2]
            V[3 * 7 + base_col + 1, col] = v[3]
            
            # Map w to global W
            # C2_block is at C[0:2, 2*block_idx : 2*block_idx+2]
            # Same mapping as before
            W[2*block_idx * 2 + 0, col] = w[0]
            W[(2*block_idx+1) * 2 + 0, col] = w[1]
            W[2*block_idx * 2 + 1, col] = w[2]
            W[(2*block_idx+1) * 2 + 1, col] = w[3]
            
            col += 1
            
    # Last block: 2x2x1
    terms = standard_2x2x1()
    for u, v, w in terms:
        U[2, col] = u[0]
        U[3, col] = u[1]
        U[6, col] = u[2]
        U[7, col] = u[3]
        base_col = 6
        V[2 * 7 + base_col, col] = v[0]
        V[3 * 7 + base_col, col] = v[1]
        W[6 * 2 + 0, col] = w[0]
        W[6 * 2 + 1, col] = w[1]
        col += 1
        
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
