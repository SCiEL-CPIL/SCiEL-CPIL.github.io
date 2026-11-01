"""Matrix multiplication 2x2x10 via Kronecker product of Strassen (rank 7) and identity (rank 5)."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For the Kronecker product approach with 2x2x2 ⊗ 1x1x5:
    # The 2x2x10 multiplication can be seen as 5 independent 2x2x2 multiplications
    # (one for each pair of columns in the 2x10 matrix)
    
    # Actually, let's think about this more carefully.
    # C = A * B where A is 2x2 and B is 2x10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # For 2x2x10: indices i in {0,1}, j in {0,1}, k in {0,...,9}
    
    # We can decompose this as: for each k, we have a 2x2x1 "slice"
    # But that's just schoolbook.
    
    # Better approach: Group the 10 columns into 5 pairs.
    # For columns (2s, 2s+1) where s = 0,1,2,3,4:
    # C[:, 2s:2s+2] = A * B[:, 2s:2s+2]
    # This is a 2x2 * 2x2 multiplication for each s.
    
    # So the total rank is 5 * rank(2x2x2) = 5 * 7 = 35 using Strassen.
    
    # Let's construct the Strassen decomposition for 2x2x2:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the Kronecker product structure:
    # The 2x2x10 tensor = sum over s=0..4 of (2x2x2 tensor for block s)
    
    # U matrix: shape (n*m, rank) = (4, 35)
    # V matrix: shape (m*p, rank) = (20, 35)
    # W matrix: shape (p*n, rank) = (20, 35)
    
    # For each block s (columns 2s, 2s+1), we have 7 Strassen terms.
    # Total rank = 5 * 7 = 35
    
    rank = 35
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen U factors (for A part, shape 4x7):
    # A indices: A11=0, A12=1, A21=2, A22=3
    # M1: A11+A22 -> [1, 0, 0, 1]
    # M2: A21+A22 -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: A11+A12 -> [1, 1, 0, 0]
    # M6: A21-A11 -> [-1, 0, 1, 0]
    # M7: A12-A22 -> [0, 1, 0, -1]
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # Strassen V factors (for B part, shape 4x7):
    # B indices: B11=0, B12=1, B21=2, B22=3
    # M1: B11+B22 -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B21-B22 -> [0, 0, 1, -1]
    # M4: B12-B11 -> [-1, 1, 0, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: B11+B12 -> [1, 1, 0, 0]
    # M7: B21+B22 -> [0, 0, 1, 1]
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # Strassen W factors (for C part, shape 4x7):
    # C indices: C11=0, C12=1, C21=2, C22=3
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ], dtype=float)
    
    # Now construct the full 2x2x10 decomposition
    # For each block s = 0,1,2,3,4:
    #   Columns in B: 2s, 2s+1 -> indices in V: j*p + k where j=0,1 and k=2s,2s+1
    #   Columns in C: 2s, 2s+1 -> indices in W: k*n + i where k=2s,2s+1 and i=0,1
    
    for s in range(5):
        for t in range(7):
            col = s * 7 + t
            
            # U factor: same for all blocks
            U[:, col] = U_strassen[:, t]
            
            # V factor: for block s, B columns are 2s and 2s+1
            # B[j, 2s] -> V[j*p + 2s, col]
            # B[j, 2s+1] -> V[j*p + 2s+1, col]
            V[0 * p + 2 * s, col] = V_strassen[0, t]    # B11
            V[0 * p + 2 * s + 1, col] = V_strassen[1, t]  # B12
            V[1 * p + 2 * s, col] = V_strassen[2, t]    # B21
            V[1 * p + 2 * s + 1, col] = V_strassen[3, t]  # B22
            
            # W factor: for block s, C columns are 2s and 2s+1
            # C[i, 2s] -> W[(2s)*n + i, col]
            # C[i, 2s+1] -> W[(2s+1)*n + i, col]
            W[2 * s * n + 0, col] = W_strassen[0, t]    # C11
            W[2 * s * n + 1, col] = W_strassen[1, t]    # C12
            W[(2 * s + 1) * n + 0, col] = W_strassen[2, t]  # C21
            W[(2 * s + 1) * n + 1, col] = W_strassen[3, t]  # C22
    
    # Verify the decomposition
    # The multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Let's verify with a random test
    
    if time.time() < deadline:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for c in range(rank):
                    C_computed[i, k] += U[i * m + 0, c] * 0  # placeholder
                # Actually, let's compute properly
                val = 0
                for j in range(m):
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
