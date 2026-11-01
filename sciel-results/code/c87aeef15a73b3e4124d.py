"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c] for all a,b,c."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=1e-9)

    def strassen_decomposition():
        """
        Apply Strassen's algorithm to 2x2 blocks of the 2x14 matrix.
        14 columns = 7 blocks of 2x2.
        Each 2x2 block multiplication uses rank 7 (Strassen).
        Total rank = 7 * 7 = 49.
        """
        rank = 7 * 7  # 49
        U = np.zeros((size_a, rank), dtype=np.float64)
        V = np.zeros((size_b, rank), dtype=np.float64)
        W = np.zeros((size_c, rank), dtype=np.float64)
        
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A11 - A22)(B12 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B11 - B21)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Output:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each block b (0 to 6), columns 2*b and 2*b+1 of B
        # A is 2x2: A[i][j] where i is row (0,1), j is col (0,1)
        # B is 2x2: B[j][k] where j is row (0,1), k is col (0,1)
        # C is 2x2: C[i][k] where i is row (0,1), k is col (0,1)
        
        # Indices in the full matrices:
        # A[i*m + j] = A[i*2 + j]
        # B[j*p + k] = B[j*14 + k]
        # C[k*n + i] = C[k*2 + i]
        
        # For block b, B's columns are 2*b and 2*b+1
        # So B[j*14 + 2*b + k] for k in {0, 1}
        
        # Strassen products for block b:
        # We need to set up U, V, W such that for each term in the Strassen decomposition,
        # we have a rank-1 contribution.
        
        # Let's define the 7 Strassen terms for a generic 2x2 block:
        # For term t (0..6), we have:
        # U_a[t] * V_b[t] * W_c[t] contributes to C[i][k]
        
        # The Strassen algorithm computes:
        # M1 = (A11 - A22)(B12 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B11 - B21)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # And:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each block b, we have 7 terms.
        # For term t in block b, the global index is r = b*7 + t
        
        # Let's build U, V, W for each block and term.
        
        # For block b, term t:
        # U[i*2 + j, r] should be the coefficient of A[i*2+j] in the first factor of M_t
        # V[j*14 + 2*b + k, r] should be the coefficient of B[j*14 + 2*b + k] in the second factor of M_t
        # W[k*2 + i, r] should be the coefficient of C[k*2 + i] in the output
        
        # This is getting complex. Let me use a more direct approach.
        
        # Actually, let's think of it differently.
        # The standard Strassen decomposition for 2x2 gives rank 7.
        # We can explicitly write out the 7 rank-1 terms.
        
        # For a single 2x2 block at position b (columns 2b, 2b+1 of B):
        # The 7 terms are:
        
        # Term 0: M1 = (A11 - A22)(B12 + B22)
        # U: A11 -> +1, A22 -> -1
        # V: B12 -> +1, B22 -> +1
        # W: C11 -> +1, C22 -> +1
        
        # Term 1: M2 = (A21 + A22)B11
        # U: A21 -> +1, A22 -> +1
        # V: B11 -> +1
        # W: C21 -> +1, C22 -> -1
        
        # Term 2: M3 = A11(B11 - B21)
        # U: A11 -> +1
        # V: B11 -> +1, B21 -> -1
        # W: C12 -> +1, C22 -> +1
        
        # Term 3: M4 = A22(B22 - B12)
        # U: A22 -> +1
        # V: B22 -> +1, B12 -> -1
        # W: C11 -> +1, C21 -> +1
        
        # Term 4: M5 = (A11 + A12)B22
        # U: A11 -> +1, A12 -> +1
        # V: B22 -> +1
        # W: C11 -> -1, C12 -> +1
        
        # Term 5: M6 = (A21 - A11)(B11 + B12)
        # U: A21 -> +1, A11 -> -1
        # V: B11 -> +1, B12 -> +1
        # W: C22 -> +1
        
        # Term 6: M7 = (A12 - A22)(B21 + B22)
        # U: A12 -> +1, A22 -> -1
        # V: B21 -> +1, B22 -> +1
        # W: C11 -> +1
        
        # Let's verify these sum correctly:
        # C11 = M1 + M4 - M5 + M7
        #   = (A11-A22)(B12+B22) + A22(B22-B12) - (A11+A12)B22 + (A12-A22)(B21+B22)
        #   = A11B12 + A11B22 - A22B12 - A22B22 + A22B22 - A22B12 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        #   = A11B12 + A11B22 - 2*A22B12 - A22B22 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        #   = A11B12 - 2*A22B12 - 2*A22B22 + A12B21 - A22B21
        #   This doesn't look right. Let me re-derive.
        
        # Standard Strassen:
        # M1 = (A11 - A22)(B12 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B11 - B21)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just implement it directly by building the tensors.
        
        for b in range(7):
            for t in range(7):
                r = b * 7 + t
                
                if t == 0:  # M1 = (A11 - A22)(B12 + B22)
                    # A11 = A[0*2+0] = A[0], A22 = A[1*2+1] = A[3]
                    # B12 = B[0*14+2*b+1] = B[2*b+1], B22 = B[1*14+2*b+1] = B[14+2*b+1]
                    # C11 = C[0*2+0] = C[0], C22 = C[1*2+1] = C[3]
                    U[0, r] += 1
                    U[3, r] += -1
                    V[2*b+1, r] += 1
                    V[14+2*b+1, r] += 1
                    W[0, r] += 1
                    W[3, r] += 1
                elif t == 1:  # M2 = (A21 + A22)B11
                    # A21 = A[1*2+0] = A[2], A22 = A[3]
                    # B11 = B[0*14+2*b+0] = B[2*b]
                    # C21 = C[0*2+1] = C[1], C22 = C[3]
                    U[2, r] += 1
                    U[3, r] += 1
                    V[2*b, r] += 1
                    W[1, r] += 1
                    W[3, r] += -1
                elif t == 2:  # M3 = A11(B11 - B21)
                    # A11 = A[0]
                    # B11 = B[2*b], B21 = B[14+2*b]
                    # C12 = C[1*2+0] = C[2], C22 = C[3]
                    U[0, r] += 1
                    V[2*b, r] += 1
                    V[14+2*b, r] += -1
                    W[2, r] += 1
                    W[3, r] += 1
                elif t == 3:  # M4 = A22(B22 - B12)
                    # A22 = A[3]
                    # B22 = B[14+2*b+1], B12 = B[2*b+1]
                    # C11 = C[0], C21 = C[1]
                    U[3, r] += 1
                    V[14+2*b+1, r] += 1
                    V[2*b+1, r] += -1
                    W[0, r] += 1
                    W[1, r] += 1
                elif t == 4:  # M5 = (A11 + A12)B22
                    # A11 = A[0], A12 = A[1]
                    # B22 = B[14+2*b+1]
                    # C11 = C[0], C12 = C[2]
                    U[0, r] += 1
                    U[1, r] += 1
                    V[14+2*b+1, r] += 1
                    W[0, r] += -1
                    W[2, r] += 1
                elif t == 5:  # M6 = (A21 - A11)(B11 + B12)
                    # A21 = A[2], A11 = A[0]
                    # B11 = B[2*b], B12 = B[2*b+1]
                    # C22 = C[3]
                    U[2, r] += 1
                    U[0, r] += -1
                    V[2*b, r] += 1
                    V[2*b+1, r] += 1
                    W[3, r] += 1
                elif t == 6:  # M7 = (A12 - A22)(B21 + B22)
                    # A12 = A[1], A22 = A[3]
                    # B21 = B[14+2*b], B22 = B[14+2*b+1]
                    # C11 = C[0]
                    U[1, r] += 1
                    U[3, r] += -1
                    V[14+2*b, r] += 1
                    V[14+2*b+1, r] += 1
                    W[0, r] += 1
        
        # Verify the decomposition
        if verify(U, V, W):
            return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try Strassen-based decomposition (rank 49)
    result = strassen_decomposition()
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to Schoolbook Rank 56
    rank = n * m * p
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
