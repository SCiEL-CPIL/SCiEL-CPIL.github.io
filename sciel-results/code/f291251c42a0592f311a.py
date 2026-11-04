"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = 4, 4, 7
SCHOOLBOOK_RANK = N * M * P  # 112
_TARGET_RANK = 97  # 49 (Strassen 4x4x4) + 48 (schoolbook 4x4x3)


def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    rank = SCHOOLBOOK_RANK
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]

    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    n, m, p = N, M, P
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matmul (Strassen)."""
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: i*m+j where m=2, so: a=0, b=1, c=2, d=3
    # V indices: j*p+k where p=2, so: e=0, f=1, g=2, h=3
    # W indices: k*n+i where n=2, so: C00=0 (i=0,k=0), C01=1 (i=0,k=1), C10=2 (i=1,k=0), C11=3 (i=1,k=1)
    
    # U[a,r], V[e,r], W[C00,r] etc.
    # M1: (a+d)(e+h) -> U[0,r]=1,U[3,r]=1; V[0,r]=1,V[3,r]=1; W[0,r]=1,W[3,r]=1
    # M2: (c+d)e -> U[2,r]=1,U[3,r]=1; V[0,r]=1; W[2,r]=1,W[3,r]=-1
    # M3: a(f-h) -> U[0,r]=1; V[1,r]=1,V[3,r]=-1; W[1,r]=1
    # M4: d(g-e) -> U[3,r]=1; V[2,r]=1,V[0,r]=-1; W[0,r]=1,W[2,r]=1
    # M5: (a+b)h -> U[0,r]=1,U[1,r]=1; V[3,r]=1; W[0,r]=-1,W[1,r]=1
    # M6: (c-a)(e+f) -> U[2,r]=1,U[0,r]=-1; V[0,r]=1,V[1,r]=1; W[3,r]=1
    # M7: (b-d)(g+h) -> U[1,r]=1,U[3,r]=-1; V[2,r]=1,V[3,r]=1; W[0,r]=1
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _tensor_product_decomp(U1, V1, W1, U2, V2, W2):
    """
    Tensor product of two matmul decompositions.
    If T1 is a decomposition for n1 x m1 x p1 and T2 for n2 x m2 x p2,
    then T1 ⊗ T2 is a decomposition for (n1*n2) x (m1*m2) x (p1*p2).
    
    U1 has shape (n1*m1, r1), V1 has shape (m1*p1, r1), W1 has shape (p1*n1, r1)
    U2 has shape (n2*m2, r2), V2 has shape (m2*p2, r2), W2 has shape (p2*n2, r2)
    
    Result: U has shape (n1*n2*m1*m2, r1*r2) = ((n1*m1)*(n2*m2), r1*r2)
    V has shape (m1*p1*m2*p2, r1*r2) = ((m1*p1)*(m2*p2), r1*r2)
    W has shape (p1*n1*p2*n2, r1*r2) = ((p1*n1)*(p2*n2), r1*r2)
    
    Indexing: 
    U[i1*m1+j1, i2*m2+j2, r1,r2] -> U_new[(i1*n2+i2)*m1*m2 + j1*m2+j2, r1*r2+r2]
    V[j1*p1+k1, j2*p2+k2, r1,r2] -> V_new[(j1*m2+j2)*p1*p2 + k1*p2+k2, r1*r2+r2]  
    W[k1*n1+i1, k2*n2+i2, r1,r2] -> W_new[(k1*n2+k2)*n1*n2 + i1*n2+i2, r1*r2+r2]
    """
    n1, m1, p1, r1 = U1.shape[0] // U1.shape[1] if False else None, None, None, U1.shape[1]
    # Let's infer dimensions
    # U1 shape: (n1*m1, r1)
    # We need to know n1, m1, p1 from context
    
    # This function is too complex to generalize without knowing dims. Let's hardcode for our case.
    raise NotImplementedError("Use specific constructions instead")


def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4x4 matmul using Strassen.
    View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    Apply Strassen at the block level (7 block multiplications),
    each requiring 7 scalar multiplications (Strassen 2x2x2).
    Total rank = 7 * 7 = 49.
    """
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    # Strassen 2x2x2 decomposition (indices for 2x2 matrices)
    # U_s[i*2+j, r], V_s[j*2+k, r], W_s[k*2+i, r]
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block structure for 4x4:
    # A11 = A[0:2, 0:2] -> U rows: 0,1,4,5
    # A12 = A[0:2, 2:4] -> U rows: 2,3,6,7
    # A21 = A[2:4, 0:2] -> U rows: 8,9,12,13
    # A22 = A[2:4, 2:4] -> U rows: 10,11,14,15
    # B11 = B[0:2, 0:2] -> V rows: 0,1,4,5
    # B12 = B[0:2, 2:4] -> V rows: 2,3,6,7
    # B21 = B[2:4, 0:2] -> V rows: 8,9,12,13
    # B22 = B[2:4, 2:4] -> V rows: 10,11,14,15
    # C11 = C[0:2, 0:2] -> W rows: 0,1,4,5
    # C12 = C[0:2, 2:4] -> W rows: 2,3,6,7
    # C21 = C[2:4, 0:2] -> W rows: 8,9,12,13
    # C22 = C[2:4, 2:4] -> W rows: 10,11,14,15
    
    # Block index mapping for 2x2 submatrices:
    # For a 2x2 block at position (bi, bj) in the 4x4 matrix:
    #   U row indices: bi*4 + bj*2 + {0,1,2,3} -> bi*4 + bj*2 + [0,1,2,3]
    #   Actually: row = i*4+j where i = bi*2+{0,1}, j = bj*2+{0,1}
    #   So rows are: bi*4 + bj*2 + 0, bi*4 + bj*2 + 1, bi*4 + bj*2 + 4, bi*4 + bj*2 + 5
    #   Wait, that's not right. Let me be more careful.
    #   i ranges over {bi*2, bi*2+1}, j ranges over {bj*2, bj*2+1}
    #   U index = i*4 + j
    #   So U indices are:
    #     (bi*2)*4 + bj*2, (bi*2)*4 + bj*2+1, (bi*2+1)*4 + bj*2, (bi*2+1)*4 + bj*2+1
    #   = bi*8 + bj*2, bi*8 + bj*2+1, bi*8 + 4 + bj*2, bi*8 + 4 + bj*2+1
    
    def get_U_indices(bi, bj):
        """Get U row indices for block (bi, bj) of a 4x4 matrix."""
        rows = []
        for di in range(2):
            for dj in range(2):
                i = bi * 2 + di
                j = bj * 2 + dj
                rows.append(i * 4 + j)
        return rows
    
    def get_V_indices(bi, bj):
        """Get V row indices for block (bi, bj) of a 4x4 matrix."""
        rows = []
        for di in range(2):
            for dj in range(2):
                j = bi * 2 + di
                k = bj * 2 + dj
                rows.append(j * 4 + k)
        return rows
    
    def get_W_indices(bi, bj):
        """Get W row indices for block (bi, bj) of a 4x4 matrix."""
        # W index = k*n + i where k is col, i is row
        # Block (bi, bj) means rows bi*2..bi*2+1, cols bj*2..bj*2+1
        rows = []
        for dk in range(2):
            for di in range(2):
                k = bj * 2 + dk
                i = bi * 2 + di
                rows.append(k * 4 + i)
        return rows
    
    # Strassen block-level formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block-level Strassen term t (0..6) and inner Strassen term s (0..6):
    #   rank index r = t * 7 + s
    
    # Define the block-level Strassen terms:
    # Each term has: A_coeffs (dict from block to coefficient), B_coeffs, C_coeffs
    # For U: coefficient for each A block
    # For V: coefficient for each B block  
    # For W: coefficient for each C block
    
    # M1: A11*A22 both +1, B11*B22 both +1, C11*C22 both +1
    # M2: A21*A22 both +1, B11 +1, C21*C22: +1, -1
    # M3: A11 +1, B12-B22: +1,-1, C12 +1
    # M4: A22 +1, B21-B11: +1,-1, C11*C21: +1,+1
    # M5: A11*A12 both +1, B22 +1, C11*C12: -1,+1
    # M6: A21-A11: +1,-1, B11*B12 both +1, C22 +1
    # M7: A12-A22: +1,-1, B21*B22 both +1, C11 +1
    
    block_strassen = [
        # (A_blocks, B_blocks, C_blocks) where each is dict {block_idx: coeff}
        # Block indices: 0=A11, 1=A12, 2=A21, 3=A22 for A
        #                0=B11, 1=B12, 2=B21, 3=B22 for B
        #                0=C11, 1=C12, 2=C21, 3=C22 for C
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),           # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),                 # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),                        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),                 # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),                 # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),                 # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),                 # M7
    ]
    
    for t, (A_coeff, B_coeff, C_coeff) in enumerate(block_strassen):
        for s in range(7):
            r = t * 7 + s
            
            # For U: for each A block with coefficient, add U_s values scaled
            for block_idx, coeff in A_coeff.items():
                bi, bj = divmod(block_idx, 2)  # bi = block_idx // 2, bj = block_idx % 2
                u_indices = get_U_indices(bi, bj)
                for local_idx, u_row in enumerate(u_indices):
                    U[u_row, r] += coeff * U_s[local_idx, s]
            
            # For V: for each B block with coefficient
            for block_idx, coeff in B_coeff.items():
                bi, bj = divmod(block_idx, 2)
                v_indices = get_V_indices(bi, bj)
                for local_idx, v_row in enumerate(v_indices):
                    V[v_row, r] += coeff * V_s[local_idx, s]
            
            # For W: for each C block with coefficient
            for block_idx, coeff in C_coeff.items():
                bi, bj = divmod(block_idx, 2)
                w_indices = get_W_indices(bi, bj)
                for local_idx, w_row in enumerate(w_indices):
                    W[w_row, r] += coeff * W_s[local_idx, s]
    
    return U.tolist(), V.tolist(), W.tolist()


def _construct_4x4x7():
    """
    Construct a rank-97 decomposition for 4x4x7.
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    
    A*B1: 4x4x4, rank 49 (Strassen)
    A*B2: 4x4x3, rank 48 (schoolbook)
    Total rank: 97
    """
    # Get 4x4x4 Strassen decomposition
    U44, V44, W44 = _construct_4x4x4_strassen()
    
    # Build 4x4x3 schoolbook decomposition
    # A is 4x4, B is 4x3, C is 4x3
    # U shape: (16, 48), V shape: (12, 48), W shape: (12, 48)
    U43 = [[0.0] * 48 for _ in range(16)]
    V43 = [[0.0] * 48 for _ in range(12)]
    W43 = [[0.0] * 48 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[i * 4 + j][col] = 1.0
                V43[j * 3 + k][col] = 1.0
                W43[k * 4 + i][col] = 1.0
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    U = [row[:] for row in U44]  # copy
    V = [row[:] for row in V44]
    W = [row[:] for row in W44]
    
    for i in range(16):
        for col in range(48):
            U[i].append(U43[i][col])
            V[i].append(V43[i][col]) if i < 12 else None
            W[i].append(W43[i][col]) if i < 12 else None
    
    # Fix V and W for rows >= 12 (they don't exist in 4x4x3)
    for i in range(12, 16):
        V[i] = [0.0] * 48
        W[i] = [0.0] * 48
    
    return U, V, W


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # For now, return schoolbook decomposition.
    # TODO: Implement faster algorithms if time permits.
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}
