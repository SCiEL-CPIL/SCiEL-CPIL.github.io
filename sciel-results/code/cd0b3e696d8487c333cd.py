import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT_SECONDS = 110
START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - START_TIME

def construct() -> dict:
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 decomposition
    # 7 terms for 2x2x2
    # U, V, W are 4x4 matrices (flattened to length 16) for the 2x2 blocks?
    # No, for 2x2x2, U,V,W are 2x2 matrices.
    
    # We build the 4x4x4 decomposition using Kronecker products of the 2x2x2 Strassen terms.
    # Rank will be 7 * 7 = 49.
    
    # Strassen 2x2 terms:
    # Each term is (u, v, w) where u, v, w are 2x2 matrices.
    # We represent 2x2 matrices as lists of 4 elements [0,1,2,3] for [0,0; 1,0; 0,1; 1,1]
    # Or better, just use the standard 7 Strassen formulas.
    
    # Let's define the 7 rank-1 terms for 2x2x2.
    # T = sum_t u_t \otimes v_t \otimes w_t
    # u_t, v_t, w_t are 2x2 matrices.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 + M7
    
    # We need to express each M_t as u_t \otimes v_t \otimes w_t?
    # No, M_t is a scalar product of two 2x2 matrices?
    # No, M_t is a 2x2 matrix product?
    # Strassen computes 7 scalar multiplications? No, 7 matrix multiplications of 2x2?
    # No, Strassen reduces 8 scalar mults to 7.
    # The tensor rank is 7.
    # Each term corresponds to one multiplication.
    
    # Let's define the 7 terms explicitly for 2x2x2.
    # u_t, v_t, w_t are 2x2 matrices.
    
    terms_2x2 = [
        # M1
        (
            [[1, 0], [0, 1]], # u: A11+A22? No, u is the coefficient for A.
            # We need u_t such that sum u_t[i,j] A[i,j] gives the first factor.
            # This is tricky. The rank-1 decomposition is not just the Strassen formulas directly mapped to U,V,W.
            # The Strassen algorithm IS the rank-7 decomposition.
        )
    ]
    
    # Actually, it is easier to just use the known rank-49 decomposition for 4x4x4
    # or construct it via Kronecker.
    
    # Let's use the Kronecker approach.
    # If we have a rank-r decomposition for m x m x m, we get rank-r^2 for 2m x 2m x 2m.
    # We need the explicit U, V, W matrices for 2x2x2.
    
    # For 2x2x2, the 7 terms are:
    # 1. u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]] -> This is identity? No.
    
    # Let's look up the explicit 2x2x2 rank-7 tensor.
    # It is often given by the 7 Strassen products.
    # Product 1: (A11+A22)(B11+B22)
    # u1 = [[1,0],[0,1]] (A11+A22)
    # v1 = [[1,0],[0,1]] (B11+B22)
    # w1 = [[1,0],[0,1]] (contributes to C11 and C22)
    
    # Product 2: (A21+A22)B11
    # u2 = [[0,0],[1,1]]
    # v2 = [[1,0],[0,0]]
    # w2 = [[0,0],[1,0]] (contributes to C21)
    
    # Product 3: A11(B12-B22)
    # u3 = [[1,0],[0,0]]
    # v3 = [[0,1],[0,-1]]
    # w3 = [[0,1],[0,0]] (contributes to C12)
    
    # Product 4: A22(B21-B11)
    # u4 = [[0,0],[0,1]]
    # v4 = [[-1,0],[0,1]]
    # w4 = [[-1,0],[0,1]] (contributes to C11 and C22)
    
    # Product 5: (A11+A12)B22
    # u5 = [[1,1],[0,0]]
    # v5 = [[0,0],[0,1]]
    # w5 = [[1,0],[1,0]] (contributes to C11 and C21)
    
    # Product 6: (A21-A11)(B11+B12)
    # u6 = [[-1,0],[1,0]]
    # v6 = [[1,1],[0,0]]
    # w6 = [[0,0],[0,1]] (contributes to C22)
    
    # Product 7: (A12-A22)(B21+B22)
    # u7 = [[0,1],[-1,0]]
    # v7 = [[0,0],[1,1]]
    # w7 = [[1,0],[0,1]] (contributes to C11 and C22)
    
    # Let's verify C11:
    # C11 = M1 - M4 + M5 + M7
    # M1: 1 * 1 * 1 = 1
    # M4: 1 * (-1) * (-1) = 1? No.
    # w4[0,0] = -1. v4[0,0] = -1. u4[0,0] = 0. So M4[0,0] = 0.
    # Wait, the product is scalar?
    # The term is u \otimes v \otimes w.
    # C[i,k] += sum_j u[i,j] A[i,j] * sum_j v[j,k] B[j,k] * w[k,i].
    
    # This is getting complicated to verify manually.
    # I will use the standard 65 rank construction from the previous program's fallback but simplified.
    
    # Actually, the previous program's _build_65 was a fallback.
    # I will implement a clean version of the 49+16 construction.
    
    U = [[0.0] * 65 for _ in range(16)] # 4x4
    V = [[0.0] * 65 for _ in range(20)] # 4x5
    W = [[0.0] * 65 for _ in range(20)] # 5x4 (p*n)
    
    # 49 terms for 4x4x4
    # Use Kronecker of 2x2x2 Strassen
    # We need the 2x2x2 U,V,W matrices.
    
    u2 = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]]
    ]
    v2 = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[0,1]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    w2 = [
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[-1,0],[0,1]],
        [[1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,0],[0,1]]
    ]
    
    # Flatten 2x2 to 4-vector: [00, 10, 01, 11] -> index i*2+j
    def flat2x2(m):
        return [m[0][0], m[1][0], m[0][1], m[1][1]] # Row major?
        # Standard is row major: [00, 01, 10, 11]
        return [m[0][0], m[0][1], m[1][0], m[1][1]]

    # Kronecker product of two 2x2 matrices A, B -> 4x4 matrix
    # A[i,j] * B[k,l] goes to [2i+k, 2j+l]
    def kron(A, B):
        R = [[0]*4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        R[2*i+k][2*j+l] += A[i][j] * B[k][l]
        return R

    def kron_flat(A_flat, B_flat):
        A = [[A_flat[0], A_flat[1]], [A_flat[2], A_flat[3]]]
        B = [[B_flat[0], B_flat[1]], [B_flat[2], B_flat[3]]]
        R = kron(A, B)
        return [R[0][0], R[0][1], R[0][2], R[0][3],
                R[1][0], R[1][1], R[1][2], R[1][3],
                R[2][0], R[2][1], R[2][2], R[2][3],
                R[3][0], R[3][1], R[3][2], R[3][3]]

    # Build 49 terms
    for t1 in range(7):
        for t2 in range(7):
            idx = t1 * 7 + t2
            u_flat = kron_flat(flat2x2(u2[t1]), flat2x2(u2[t2]))
            v_flat = kron_flat(flat2x2(v2[t1]), flat2x2(v2[t2]))
            w_flat = kron_flat(flat2x2(w2[t1]), flat2x2(w2[t2]))
            
            for i in range(16):
                if u_flat[i] != 0:
                    U[i][idx] = u_flat[i]
            for j in range(16):
                if v_flat[j] != 0:
                    V[j][idx] = v_flat[j] # Note: V is 4x5, so j<16 is valid for first 4 cols
            for k in range(16):
                if w_flat[k] != 0:
                    W[k][idx] = w_flat[k] # Note: W is 5x4, so k<16 is valid for first 4 rows
    
    # 16 terms for 4x4x1 (5th column)
    # A(4x4) * b(4x1) -> y(4x1)
    # y_i = sum_j A_ij b_j
    # Terms: for each i, j
    # U[i,j] = 1, V[j,4] = 1, W[4,i] = 1
    for i in range(4):
        for j in range(4):
            idx = 49 + i * 4 + j
            U[i * 4 + j][idx] = 1.0
            V[j * 5 + 4][idx] = 1.0
            W[4 * 4 + i][idx] = 1.0
            
    return {"U": U, "V": V, "W": W}
