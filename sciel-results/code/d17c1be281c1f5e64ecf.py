"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j, :]
                vk = V[j * p + k, :]
                wk = W[k * n + i, :]
                val = np.dot(ui, vk * wk)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) where U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+c)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U[i*2+j, c] for i,j in 0,1 and c in 0..6
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+g): U contributions for a and d, V for e and g, W for all C entries
    # a is at U[0,0], d is at U[1,1]
    U[0, 0] = 1  # a
    U[1, 0] = 1  # d  (wait, 1*2+1 = 3)
    # Let me re-index: U[i*2+j, c]
    # a = A[0,0] -> U[0, c]
    # b = A[0,1] -> U[1, c]
    # c = A[1,0] -> U[2, c]
    # d = A[1,1] -> U[3, c]
    
    # B[0,0]=e -> V[0,c], B[0,1]=f -> V[1,c], B[1,0]=g -> V[2,c], B[1,1]=h -> V[3,c]
    
    # C[0,0] -> W[0,c], C[0,1] -> W[1,c], C[1,0] -> W[2,c], C[1,1] -> W[3,c]
    
    # M1 = (a+d)(e+g)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[2, 0] = 1  # g
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # M3 = a(g-h)
    U[0, 2] = 1  # a
    V[2, 2] = 1  # g
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = d(f-e)
    U[3, 3] = 1  # d
    V[1, 3] = 1  # f
    V[0, 3] = -1 # e
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (a+c)h
    U[0, 4] = 1  # a
    U[2, 4] = 1  # c
    V[3, 4] = 1  # h
    W[1, 4] = 1  # C[0,1]
    W[0, 4] = -1 # C[0,0]
    
    # M6 = (c-a)(e+g)
    U[2, 5] = 1  # c
    U[0, 5] = -1 # a
    V[0, 5] = 1  # e
    V[2, 5] = 1  # g
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    start_time = time.time()
    
    # Strategy: Use recursive Strassen-style decomposition
    # 4x4x6 = (2x2x3) x (2x2x2) via Kronecker product structure
    # Actually, let's use the approach from the knowledge:
    # 4x4*4x4 via recursive Strassen (rank 49) + 4x4*4x2 via Strassen blocks (rank 28) = 77
    
    # Better: decompose 4x4*4x6 as:
    # Split the 6 columns into 2 groups of 3: 4x4*4x3 + 4x4*4x3
    # Or split into 4x4*4x2 + 4x4*4x2 + 4x4*4x2
    # Or use the 2x2x3 base case with rank 11
    
    # Let's use the Kronecker product approach:
    # 4x4x6 = tensor product of (2x2x3) with (2x2x2)
    # Rank = 11 * 7 = 77
    
    # First, build the 2x2x3 decomposition (rank 11)
    # C = A * B where A is 2x2, B is 2x3
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h  
    # C[0,2] = a*i + b*j
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    # C[1,2] = c*i + d*j
    
    # Use Strassen for first 2 columns (rank 7) + schoolbook for 3rd column (rank 4)
    U_223 = np.zeros((4, 11), dtype=np.float64)
    V_223 = np.zeros((6, 11), dtype=np.float64)
    W_223 = np.zeros((6, 11), dtype=np.float64)
    
    # Strassen part (columns 0,1 of B, i.e., e,f,g,h)
    S_U, S_V, S_W = _strassen_2x2x2()
    U_223[:, 0:7] = S_U
    V_223[0:4, 0:7] = S_V  # e,f,g,h map to V[0:4, :]
    V_223[4:6, 0:7] = 0    # i,j don't participate
    W_223[0:4, 0:7] = S_W  # C[0,0],C[0,1],C[1,0],C[1,1]
    W_223[4:6, 0:7] = 0    # C[*,2] don't come from Strassen
    
    # Schoolbook for 3rd column (i,j)
    # C[0,2] = a*i + b*j
    # C[1,2] = c*i + d*j
    U_223[0, 7] = 1  # a
    U_223[1, 8] = 1  # b
    U_223[2, 9] = 1  # c
    U_223[3, 10] = 1 # d
    V_223[4, 7] = 1  # i (B[0,2])
    V_223[5, 8] = 1  # j (B[1,2])
    V_223[4, 9] = 1  # i
    V_223[5, 10] = 1 # j
    W_223[4, 7] = 1  # C[0,2]
    W_223[5, 8] = 1  # C[1,2]
    W_223[4, 9] = 1  # C[0,2]
    W_223[5, 10] = 1 # C[1,2]
    
    # Verify 2x2x3
    if not _verify(U_223, V_223, W_223, 2, 2, 3):
        # Fallback: schoolbook
        U_223 = np.zeros((4, 12), dtype=np.float64)
        V_223 = np.zeros((6, 12), dtype=np.float64)
        W_223 = np.zeros((6, 12), dtype=np.float64)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_223[i*2+j, col] = 1
                    V_223[j*3+k, col] = 1
                    W_223[k*2+i, col] = 1
                    col += 1
        return _schoolbook_construct(n, m, p)
    
    # Now build the 2x2x2 decomposition (rank 7)
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # Kronecker product: 4x4x6 = (2x2x3) ⊗ (2x2x2)
    # The tensor T_{i,j,k} for 4x4x6 can be written as T_{(i1,i2),(j1,j2),(k1,k2)} 
    # but wait, the dimensions are n=4, m=4, p=6.
    # 4 = 2*2, 4 = 2*2, 6 = 3*2
    # So we decompose as:
    # i = i1*2 + i2 (i1 in 0,1; i2 in 0,1)
    # j = j1*2 + j2 (j1 in 0,1; j2 in 0,1)  
    # k = k1*2 + k2 (k1 in 0,1,2; k2 in 0,1)
    
    # The matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # With the Kronecker structure, this factorizes as:
    # T[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # where T1 is the 2x2x3 tensor and T2 is the 2x2x2 tensor
    
    # So the rank-77 decomposition is the Kronecker product of the rank-11 and rank-7 decompositions
    
    rank = 11 * 7
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((24, rank), dtype=np.float64)
    W = np.zeros((24, rank), dtype=np.float64)
    
    for c1 in range(11):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U[i, c] = U_223[i1, c1] * U_222[i2, c2]
            # i = i1*2 + i2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i * 2 + 0, c] = 0  # placeholder
            # Actually U has shape (n*m, rank) = (16, rank)
            # U[i*m + j, c] where i in 0..3, j in 0..3
            # i = i1*2 + i2, j = j1*2 + j2
            # U[i*m+j, c] = U_223[i1*2+j1, c1] * U_222[i2*2+j2, c2]
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, c] = U_223[i1 * 2 + j1, c1] * U_222[i2 * 2 + j2, c2]
            
            # V[j*p + k, c] = V_223[j1*3+k1, c1] * V_222[j2*2+k2, c2]
            # j = j1*2 + j2, k = k1*2 + k2
            for j1 in range(2):
                for k1 in range(3):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 6 + k, c] = V_223[j1 * 3 + k1, c1] * V_222[j2 * 2 + k2, c2]
            
            # W[k*n + i, c] = W_223[k1*2+i1, c1] * W_222[k2*2+i2, c2]
            for k1 in range(3):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, c] = W_223[k1 * 2 + i1, c1] * W_222[k2 * 2 + i2, c2]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to list of lists with integer/half-integer entries
        U_list = [[round(val, 6) for val in row] for row in U.tolist()]
        V_list = [[round(val, 6) for val in row] for row in V.tolist()]
        W_list = [[round(val, 6) for val in row] for row in W.tolist()]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook_construct(n, m, p)

def _schoolbook_construct(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
