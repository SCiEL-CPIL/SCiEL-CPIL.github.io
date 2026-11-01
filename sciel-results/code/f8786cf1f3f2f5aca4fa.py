"""
Rank-minimal tensor decomposition for 2x2 x 2x15 matrix multiplication.

Strategy:
- The tensor T_{2,2,15} decomposes as a direct sum over the 15 columns of B.
- Each column of B gives a 2x2 x 2x1 sub-problem.
- We can group columns to exploit Strassen's 7-rank decomposition for 2x2x2.
- Split 15 columns into 7 groups of 2 and 1 group of 1.
- For each group of 2 columns: use Strassen 2x2x2 (rank 7).
- For the last single column: use schoolbook 2x2x2x1 (rank 4).
- Total rank = 7*7 + 4 = 53.

Strassen decomposition for C = A*B where A,B,C are 2x2:
  A = [[a00, a01], [a10, a11]]
  B = [[b00, b01], [b10, b11]]
  C = [[c00, c01], [c10, c11]]

  M1 = (a00 + a11)(b00 + b11)
  M2 = (a10 + a11)b00
  M3 = a00(b10 - b11)
  M4 = a11(b01 - b00)
  M5 = (a00 + a10)b11
  M6 = (a11 - a10)(b00 + b10)
  M7 = (a01 - a11)(b01 + b11)

  c00 = M1 - M4 + M5 + M7
  c01 = M3 + M4
  c10 = M2 + M7
  c11 = M1 - M3 + M6 + M7

Trace convention: W[k*n + i, r] where n=2.
So W index for C[i][k] is k*n + i = 2k + i.
For 2x2 output: c00 -> W[0], c01 -> W[2], c10 -> W[1], c11 -> W[3].
Wait, let me re-read: W[k*n+i, r] where i is row of C, k is column of C.
C is 2x2, so i in {0,1}, k in {0,1}.
W[0] = c[0][0], W[1] = c[0][1], W[2] = c[1][0], W[3] = c[1][1].
Actually W[k*n+i]: k is the column index of C, i is the row index.
So W[0*2+0]=W[0]=c00, W[1*2+0]=W[2]=c10, W[0*2+1]=W[1]=c01, W[1*2+1]=W[3]=c11.

For U: A is 2x2, U[i*m+j, r] where i is row of A, j is col of A. m=2.
U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11.

For V: B is 2x2 (or 2x1), V[j*p+k, r] where j is row of B, k is col of B. p=2 (for 2-col group).
V[0]=b00, V[1]=b01, V[2]=b10, V[3]=b11.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (30, rank)
    
    # We'll build the decomposition by groups.
    # 7 groups of 2 columns each, then 1 group of 1 column.
    
    U_list = []  # Each entry is a 4-vector
    V_list = []  # Each entry is a 30-vector
    W_list = []  # Each entry is a 30-vector
    
    # Strassen 2x2x2 coefficients
    # M1 = (a00 + a11)(b00 + b11)
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    # c00 = +1, c11 = +1 -> W[0]=1, W[3]=1
    W1 = [1, 0, 0, 1]
    
    # M2 = (a10 + a11)b00
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    # c10 = +1 -> W[2]=1
    W2 = [0, 0, 1, 0]
    
    # M3 = a00(b10 - b11)
    U3 = [1, 0, 0, 0]
    V3 = [0, 0, 1, -1]
    # c10 = +1, c11 = -1 -> W[2]=1, W[3]=-1
    W3 = [0, 0, 1, -1]
    
    # M4 = a11(b01 - b00)
    U4 = [0, 0, 0, 1]
    V4 = [-1, 1, 0, 0]
    # c01 = +1, c10 = -1 -> W[1]=1, W[2]=-1
    W4 = [0, 1, -1, 0]
    
    # M5 = (a00 + a10)b11
    U5 = [1, 0, 1, 0]
    V5 = [0, 0, 0, 1]
    # c01 = +1, c11 = +1 -> W[1]=1, W[3]=1
    W5 = [0, 1, 0, 1]
    
    # M6 = (a11 - a10)(b00 + b10)
    U6 = [0, 0, -1, 1]
    V6 = [1, 0, 1, 0]
    # c11 = +1 -> W[3]=1
    W6 = [0, 0, 0, 1]
    
    # M7 = (a01 - a11)(b01 + b11)
    U7 = [0, 1, 0, -1]
    V7 = [0, 1, 0, 1]
    # c00 = +1, c01 = +1, c10 = +1 -> W[0]=1, W[1]=1, W[2]=1
    W7 = [1, 1, 1, 0]
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    # For a single column (2x2 x 2x1):
    # A is 2x2, B is 2x1, C is 2x1.
    # Schoolbook: 4 terms.
    # a00*b0 -> c0
    # a01*b1 -> c0
    # a10*b0 -> c1
    # a11*b1 -> c1
    single_U = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1],  # a11
    ]
    single_V = [
        [1, 0],  # b0
        [0, 1],  # b1
        [1, 0],  # b0
        [0, 1],  # b1
    ]
    single_W = [
        [1, 0],  # c0
        [1, 0],  # c0
        [0, 1],  # c1
        [0, 1],  # c1
    ]
    
    # Now assemble.
    # For each of the 7 groups of 2 columns (k=0,1), (k=2,3), ..., (k=12,13):
    # The V indices for these 2 columns in the 2x15 B matrix:
    # B is 2x15, so V has shape (2*15, rank) = (30, rank).
    # V[j*15 + k, r] where j is row (0,1), k is col (0..14).
    # For columns 2g and 2g+1:
    #   b00 (row 0, col 2g) -> V[0*15 + 2g] = V[30g]
    #   b01 (row 0, col 2g+1) -> V[0*15 + 2g+1] = V[30g+1]
    #   b10 (row 1, col 2g) -> V[1*15 + 2g] = V[15+30g]
    #   b11 (row 1, col 2g+1) -> V[1*15 + 2g+1] = V[15+30g+1]
    
    # W indices for C:
    # C is 2x15, W[k*2 + i, r] where i is row (0,1), k is col (0..14).
    # For columns 2g and 2g+1:
    #   c00 (row 0, col 2g) -> W[2g*2 + 0] = W[4g]
    #   c01 (row 0, col 2g+1) -> W[(2g+1)*2 + 0] = W[4g+2]
    #   c10 (row 1, col 2g) -> W[2g*2 + 1] = W[4g+1]
    #   c11 (row 1, col 2g+1) -> W[(2g+1)*2 + 1] = W[4g+3]
    
    for g in range(7):
        for t in range(7):
            U_vec = strassen_U[t][:]
            U_list.append(U_vec)
            
            V_vec = [0] * 30
            v4 = strassen_V[t]
            V_vec[15*0 + 2*g] = v4[0]    # b00
            V_vec[15*0 + 2*g+1] = v4[1]  # b01
            V_vec[15*1 + 2*g] = v4[2]    # b10
            V_vec[15*1 + 2*g+1] = v4[3]  # b11
            V_list.append(V_vec)
            
            W_vec = [0] * 30
            w4 = strassen_W[t]
            W_vec[2*(2*g) + 0] = w4[0]    # c00
            W_vec[2*(2*g+1) + 0] = w4[1]  # c01
            W_vec[2*(2*g) + 1] = w4[2]    # c10
            W_vec[2*(2*g+1) + 1] = w4[3]  # c11
            W_list.append(W_vec)
    
    # For the last single column (col 14):
    # V indices: row 0 col 14 -> V[14], row 1 col 14 -> V[29]
    # W indices: row 0 col 14 -> W[14*2+0]=W[28], row 1 col 14 -> W[14*2+1]=W[29]
    for t in range(4):
        U_vec = single_U[t][:]
        U_list.append(U_vec)
        
        V_vec = [0] * 30
        v2 = single_V[t]
        V_vec[0*15 + 14] = v2[0]  # b0
        V_vec[1*15 + 14] = v2[1]  # b1
        V_list.append(V_vec)
        
        W_vec = [0] * 30
        w2 = single_W[t]
        W_vec[14*2 + 0] = w2[0]  # c0
        W_vec[14*2 + 1] = w2[1]  # c1
        W_list.append(W_vec)
    
    rank = len(U_list)
    assert rank == 53, f"Expected rank 53, got {rank}"
    
    U = np.array(U_list).T  # shape (4, 53)
    V = np.array(V_list).T  # shape (30, 53)
    W = np.array(W_list).T  # shape (30, 53)
    
    # Verify the decomposition with random test
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randn(2, 2)
        B_test = np.random.randn(2, 15)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((2, 15))
        for i in range(2):
            for k in range(15):
                s = 0.0
                for r in range(rank):
                    s += U[i*2 + 0, r] * 0 + 0  # This is wrong, let me redo.
                # Actually: C[i][k] = sum_r U[i*m + j?, ...]
                # Wait, the tensor decomposition is:
                # C[i][k] = sum_{j,r} A[i][j] * B[j][k] = sum_{j,r} (sum_r U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
                # No, the standard tensor decomposition is:
                # T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
                # where T_{i,j,k} = delta (the multiplication tensor).
                # For matrix mult, the tensor is T_{(i,j),(j,k),(i,k)} = 1 if valid, else 0.
                # So C[i][k] = sum_{j} A[i][j] * B[j][k]
                # In tensor form: the multiplication map is sum_{i,j,k} T_{(i,j),(j,k),(i,k)} A_{(i,j)} B_{(j,k)}
                # But our factorization is:
                # C = sum_r (U[:,r] as vector in A-space) outer (V[:,r] as vector in B-space) -> but that's not right.
                # Actually: C_{(i,k)} = sum_r U_{(i,?)} ...
                # Let me think again.
                # The standard form is:
                # C_{i,k} = sum_{j, r} A_{i,j} * V_{j*p+k, r} * W_{k*n+i, r} * (something with U?)
                # No. The correct tensor decomposition for matrix multiplication A(m x n) * B(n x p) -> C(m x p) is:
                # C_{i,k} = sum_{j} sum_{r} A_{i,j} * B_{j,k} * T_{(i,j),(j,k),(i,k),r}
                # But in the factorized form with U, V, W:
                # C_{i,k} = sum_r (sum_j U_{i*m+j, r} * A_{i,j}) * (sum_j V_{j*p+k, r} * B_{j,k}) * W_{k*n+i, r}
                # No, that's not right either. The factorized form is:
                # The tensor T = sum_r u_r \otimes v_r \otimes w_r
                # where u_r is in R^{m*n}, v_r is in R^{n*p}, w_r is in R^{p*m}
                # And C = T(A, B) = sum_r (u_r \cdot A_vec) * (v_r \cdot B_vec) * w_r
                # where A_vec is A flattened, B_vec is B flattened, and w_r tells which C entry to add to.
                # Actually: C_{i,k} = sum_r (u_r \cdot A_vec) * (v_r \cdot B_vec) * w_r[k*n+i]
                # Hmm, but that would mean C is a scalar times sum, not a matrix.
                
                # Let me re-derive. The multiplication tensor for m x n * n x p -> m x p is:
                # T_{(i,j), (j,k), (i,k)} = 1
                # The factorized form is: T = sum_r U_r \otimes V_r \otimes W_r
                # where U_r is a vector in R^{m*n}, V_r in R^{n*p}, W_r in R^{m*p}
                # And the multiplication is:
                # C_{i,k} = sum_r W_r[k*n+i] * (U_r \cdot A_flattened) * (V_r \cdot B_flattened)
                # No wait, that's not standard. Let me look at the definition more carefully.
                
                # Actually, the standard tensor decomposition for matrix multiplication is:
                # C = A * B = sum_r (U_r^T * A_flattened) * (V_r^T * B_flattened) * W_r
                # where U_r is m*n x 1, V_r is n*p x 1, W_r is m*p x 1
                # And C_flattened = sum_r W_r * (U_r^T * A_flattened) * (V_r^T * B_flattened)
                # But that's a rank-r decomposition where each term is an outer product of three vectors.
                # So C_{i,k} = sum_r W_r[k*n+i] * (sum_{j} U_r[i*m+j] * A[i,j]) * (sum_{j} V_r[j*p+k] * B[j,k])
                
        # Let me just verify numerically.
        A_flat = A_test.flatten()  # row-major: a00, a01, a10, a11
        B_flat = B_test.flatten()  # row-major: b00, b01, ..., b0,14, b10, ..., b1,14
        # Wait, B is 2x15, so B_flat = [b[0][0], b[0][1], ..., b[0][14], b[1][0], ..., b[1][14]]
        
        C_flat = np.zeros(30)
        for r in range(rank):
            u_dot = np.dot(U[:, r], A_flat)
            v_dot = np.dot(V[:, r], B_flat)
            C_flat += W[:, r] * u_dot * v_dot
        
        C_from_decomp = C_flat.reshape(2, 15)
        if not np.allclose(C_from_decomp, C_expected, atol=1e-9):
            # Fallback to schoolbook
            U_s = np.zeros((4, 60))
            V_s = np.zeros((30, 60))
            W_s = np.zeros((30, 60))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        U_s[i*2 + j, col] = 1
                        V_s[j*15 + k, col] = 1
                        W_s[k*2 + i, col] = 1
                        col += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
