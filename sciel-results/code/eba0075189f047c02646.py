"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-99 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    1. Decompose the 4x4x9 problem into two 4x4x4 blocks and one 4x4x1 block.
    2. Use the Kronecker product of the rank-7 Strassen decomposition for 2x2x2
       to build a rank-49 decomposition for each 4x4x4 block.
    3. Use a rank-1 decomposition for the 4x4x1 block.
    4. Total rank: 49 + 49 + 1 = 99.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    deadline = time.time() + 140

    # --- Step 1: Define Strassen rank-7 decomposition for 2x2x2 ---
    # Indices for 2x2x2:
    # U: i*2+j (0..3)
    # V: j*2+k (0..3)
    # W: k*2+i (0..3)
    
    # The 7 components (u, v, w) as vectors of length 4
    strassen_2x2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00 -> C10
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # W index: k*2+i. C10 is k=1, i=0 -> index 2? 
        # Wait, W index for 2x2x2 is k*2+i.
        # C00: k=0, i=0 -> 0
        # C01: k=0, i=1 -> 1
        # C10: k=1, i=0 -> 2
        # C11: k=1, i=1 -> 3
        # Let's re-verify W indices.
        # Problem statement: W[k*n+i, r]. For 2x2x2, n=2.
        # So W index is k*2+i.
        # C00 (k=0, i=0) -> 0
        # C01 (k=0, i=1) -> 1
        # C10 (k=1, i=0) -> 2
        # C11 (k=1, i=1) -> 3
        
        # M1: C00 + C11 -> W = [1, 0, 0, 1]
        # M2: C10 -> W = [0, 0, 1, 0]
        # M3: C01 + C11? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    # Corrected Strassen components (u, v, w)
    # u: i*2+j. 00, 01, 10, 11 -> 0, 1, 2, 3
    # v: j*2+k. 00, 01, 10, 11 -> 0, 1, 2, 3
    # w: k*2+i. 00, 01, 10, 11 -> 0, 1, 2, 3 (k is row of C, i is col of C? No, i is row of A/C, k is col of B/C)
    # C index: k*n+i. n=2.
    # C00: k=0, i=0 -> 0
    # C01: k=0, i=1 -> 1
    # C10: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    
    S = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([1, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # M6
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, -1])),
    ]
    
    # --- Step 2: Build 4x4x4 decomposition using Kronecker product ---
    # Rank 7 x 7 = 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col = 0
    for i in range(7):
        for j in range(7):
            u_kron = np.kron(S[i][0], S[j][0])
            v_kron = np.kron(S[i][1], S[j][1])
            w_kron = np.kron(S[i][2], S[j][2])
            U_444[:, col] = u_kron
            V_444[:, col] = v_kron
            W_444[:, col] = w_kron
            col += 1
            
    # --- Step 3: Build 4x4x1 decomposition (rank 1) ---
    # This block corresponds to the last column of B (k=8 in global, or k=0 in local block)
    # Local indices: i in 0..3, j in 0..3, k=0.
    # U index: i*4+j
    # V index: j*1+0 = j (local p=1)
    # W index: 0*4+i = i (local n=4)
    
    U_441 = np.zeros((16, 1))
    V_441 = np.zeros((4, 1)) # Local V shape is m x p = 4 x 1
    W_441 = np.zeros((4, 1)) # Local W shape is p x n = 1 x 4? No, p*n = 4.
    
    # Wait, global V shape is (m*p, rank) = (36, rank).
    # Global W shape is (p*n, rank) = (36, rank).
    # For the 4x4x1 block, we only use specific rows of V and W.
    
    # U_441: All ones
    U_441[:, 0] = 1.0
    
    # V_441: Only rows corresponding to k=0 in the local block.
    # Local V index j*1+0 = j.
    # So V_441[j, 0] = 1 for j in 0..3.
    V_441[:, 0] = 1.0
    
    # W_441: Only rows corresponding to k=0 in the local block.
    # Local W index 0*4+i = i.
    # So W_441[i, 0] = 1 for i in 0..3.
    W_441[:, 0] = 1.0

    # --- Step 4: Assemble Global Matrices ---
    # Global Rank = 49 + 49 + 1 = 99
    total_rank = 99
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: 4x4x4 (columns 0-3 of B)
    # Global V index: j*9 + k. k in 0..3.
    # Global W index: k*4 + i. k in 0..3.
    
    for r in range(49):
        U[:, r] = U_444[:, r]
        
        # Map V_444 (index j*4+k) to V (index j*9+k)
        for j in range(4):
            for k in range(4):
                local_idx = j * 4 + k
                global_idx = j * 9 + k
                V[global_idx, r] = V_444[local_idx, r]
                
        # Map W_444 (index k*4+i) to W (index k*4+i)
        # Indices are identical for the first 16 entries
        W[0:16, r] = W_444[:, r]
        
    # Block 2: 4x4x4 (columns 4-7 of B)
    # Global V index: j*9 + k. k in 4..7.
    # Global W index: k*4 + i. k in 4..7.
    
    for r in range(49):
        U[:, 49 + r] = U_444[:, r]
        
        # Map V_444 (index j*4+k) to V (index j*9 + (k+4))
        for j in range(4):
            for k in range(4):
                local_idx = j * 4 + k
                global_idx = j * 9 + (k + 4)
                V[global_idx, 49 + r] = V_444[local_idx, r]
                
        # Map W_444 (index k*4+i) to W (index (k+4)*4+i)
        for k in range(4):
            for i in range(4):
                local_idx = k * 4 + i
                global_idx = (k + 4) * 4 + i
                W[global_idx, 49 + r] = W_444[local_idx, r]
                
    # Block 3: 4x4x1 (column 8 of B)
    # Global V index: j*9 + 8.
    # Global W index: 8*4 + i = 32 + i.
    
    for r in range(1):
        U[:, 98 + r] = U_441[:, r]
        
        # Map V_441 (index j) to V (index j*9 + 8)
        for j in range(4):
            global_idx = j * 9 + 8
            V[global_idx, 98 + r] = V_441[j, r]
            
        # Map W_441 (index i) to W (index 32 + i)
        for i in range(4):
            global_idx = 32 + i
            W[global_idx, 98 + r] = W_441[i, r]

    # --- Verification ---
    def verify(U, V, W, tol=1e-8):
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
