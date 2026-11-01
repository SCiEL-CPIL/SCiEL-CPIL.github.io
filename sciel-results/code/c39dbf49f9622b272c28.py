import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is valid."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def _build_strassen_block(n, m, p):
    """
    Build a rank-minimal decomposition for n x m x p multiplication
    assuming small dimensions where Strassen-like splitting is beneficial.
    This function is a placeholder for the specific 2x2x2 Strassen logic.
    """
    # For 2x2x2, rank is 7.
    # We will implement the specific 2x2x7 construction in the main construct function
    # by combining smaller blocks.
    pass

def _make_strassen_2x2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    Dimensions: n=2, m=2, p=2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # Strassen's 7 multiplications:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Indices for 2x2x2:
    # a_idx = i*2 + j
    # b_idx = j*2 + k
    # c_idx = k*2 + i
    
    # We need to map the 7 products to the tensor factors.
    # U[a, r], V[b, r], W[c, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A11 + A22) * (B11 + B22)
    # A11 is a=0 (i=0,j=0), A22 is a=3 (i=1,j=1)
    # B11 is b=0 (j=0,k=0), B22 is b=3 (j=1,k=1)
    # C11 is c=0 (k=0,i=0), C12 is c=1 (k=0,i=1? No, c=k*n+i. k=0,i=1 -> c=1)
    # C21 is c=2 (k=1,i=0), C22 is c=3 (k=1,i=1)
    # C = M1 - M4 - M5 + M7 (for C11)
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # Let's define the contributions to each C element.
    # C11 (c=0): M1 - M4 - M5 + M7
    # C12 (c=1): M3 + M5
    # C21 (c=2): M2 + M4
    # C22 (c=3): M1 - M2 - M3 + M6
    
    # Factors for each multiplication r:
    # r=0 (M1):
    # U: A11 (a=0) + A22 (a=3) -> U[0,0]=1, U[3,0]=1
    # V: B11 (b=0) + B22 (b=3) -> V[0,0]=1, V[3,0]=1
    # W: contributes to C11 (+1), C22 (+1) -> W[0,0]=1, W[3,0]=1
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1 (M2): (A21 + A22) * B11
    # A21 is a=2 (i=1,j=0), A22 is a=3 (i=1,j=1)
    # B11 is b=0
    # W: contributes to C21 (+1), C22 (-1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2 (M3): A11 * (B12 - B22)
    # A11 is a=0
    # B12 is b=1 (j=0,k=1), B22 is b=3
    # W: contributes to C12 (+1), C22 (-1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # r=3 (M4): A22 * (B21 - B11)
    # A22 is a=3
    # B21 is b=2 (j=1,k=0), B11 is b=0
    # W: contributes to C11 (-1), C21 (+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # r=4 (M5): (A11 + A12) * B22
    # A11 is a=0, A12 is a=1 (i=0,j=1)
    # B22 is b=3
    # W: contributes to C11 (-1), C12 (+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5 (M6): (A21 - A11) * (B11 + B12)
    # A21 is a=2, A11 is a=0
    # B11 is b=0, B12 is b=1
    # W: contributes to C22 (+1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6 (M7): (A12 - A22) * (B21 + B22)
    # A12 is a=1, A22 is a=3
    # B21 is b=2, B22 is b=3
    # W: contributes to C11 (+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _make_trivial_2x2x1_factors():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 4.
    Dimensions: n=2, m=2, p=1.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C = A B
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22  (Wait, p=1, so B is 2x1. B11, B21. C is 2x1. C11, C21)
    # Let's check indices for 2x2x1.
    # n=2, m=2, p=1.
    # a = i*2 + j. i in 0,1. j in 0,1.
    # b = j*1 + k. j in 0,1. k in 0.
    # c = k*2 + i. k in 0. i in 0,1.
    
    # Terms:
    # i=0, j=0, k=0: a=0, b=0, c=0. A11 B11 -> C11
    # i=0, j=1, k=0: a=1, b=1, c=0. A12 B21 -> C11
    # i=1, j=0, k=0: a=2, b=0, c=1. A21 B11 -> C21
    # i=1, j=1, k=0: a=3, b=1, c=1. A22 B21 -> C21
    
    # r=0: A11 B11 -> C11
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # r=1: A12 B21 -> C11
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # r=2: A21 B11 -> C21
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # r=3: A22 B21 -> C21
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    
    # Try the structural block decomposition:
    # Split m=4 into 2+2.
    # Block 1: A[0:2, 0:2] (2x2) * B[0:2, 0:7] (2x7) -> C[0:2, 0:7] (2x7)
    # Block 2: A[0:2, 2:4] (2x2) * B[2:4, 0:7] (2x7) -> C[0:2, 0:7] (2x7)
    
    # We need to decompose 2x2x7 multiplication.
    # Split p=7 into 2+2+2+1.
    # 3 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Trivial, rank 4).
    # Total rank for one 2x2x7 block: 7+7+7+4 = 25.
    # Two such blocks: 50.
    
    # Let's build the 2x2x7 decomposition first.
    # U_2x2x7: (4, 25)
    # V_2x2x7: (4, 25)  -- Wait, V dimension is m*p. For 2x2x7, m=2, p=7, so V is (14, 25)?
    # No, in the sub-problem, the dimensions are n_sub=2, m_sub=2, p_sub=7.
    # U_sub shape: (n_sub*m_sub, rank) = (4, 25)
    # V_sub shape: (m_sub*p_sub, rank) = (14, 25)
    # W_sub shape: (p_sub*n_sub, rank) = (14, 25)
    
    # Let's construct U_sub, V_sub, W_sub for 2x2x7.
    # It consists of 4 independent sub-sub-problems corresponding to k splits.
    # k_split 0: k in {0,1} -> 2x2x2
    # k_split 1: k in {2,3} -> 2x2x2
    # k_split 2: k in {4,5} -> 2x2x2
    # k_split 3: k in {6}   -> 2x2x1
    
    U_sub = np.zeros((4, 25))
    V_sub = np.zeros((14, 25))
    W_sub = np.zeros((14, 25))
    
    r_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(U, V, W, r_off, k_start):
        U_s, V_s, W_s = _make_strassen_2x2x2_factors()
        # U_s is (4, 7). V_s is (4, 7) -- Wait, V_s in _make_strassen is (4,7) for m=2,p=2.
        # In the sub-problem 2x2x2, m=2, p=2. V dim is 4.
        # In the 2x2x7 context, the B matrix is 2x7.
        # The rows of B corresponding to this k_split are k_start and k_start+1.
        # The index in V_sub for b = j*p_sub + k.
        # Here p_sub=7.
        # For the Strassen block, the local b indices are 0..3 (j in 0,1; k_local in 0,1).
        # Global b indices: j*7 + (k_start + k_local).
        
        # U is just A, which is 2x2. Indices 0..3. Same for all k_splits.
        U[:, r_off:r_off+7] = U_s
        
        # V mapping:
        # Local b=0 (j=0, k_loc=0) -> Global b = 0*7 + k_start = k_start
        # Local b=1 (j=0, k_loc=1) -> Global b = 0*7 + k_start + 1 = k_start + 1
        # Local b=2 (j=1, k_loc=0) -> Global b = 1*7 + k_start = 7 + k_start
        # Local b=3 (j=1, k_loc=1) -> Global b = 1*7 + k_start + 1 = 7 + k_start + 1
        
        V[k_start, r_off:r_off+7] = V_s[0, :]
        V[k_start+1, r_off:r_off+7] = V_s[1, :]
        V[7+k_start, r_off:r_off+7] = V_s[2, :]
        V[7+k_start+1, r_off:r_off+7] = V_s[3, :]
        
        # W mapping:
        # Local c = k_local * 2 + i.
        # Global c = k_global * 2 + i = (k_start + k_local) * 2 + i.
        # Local c=0 (k_loc=0, i=0) -> Global c = k_start*2 + 0
        # Local c=1 (k_loc=0, i=1) -> Global c = k_start*2 + 1
        # Local c=2 (k_loc=1, i=0) -> Global c = (k_start+1)*2 + 0
        # Local c=3 (k_loc=1, i=1) -> Global c = (k_start+1)*2 + 1
        
        W[k_start*2, r_off:r_off+7] = W_s[0, :]
        W[k_start*2+1, r_off:r_off+7] = W_s[1, :]
        W[(k_start+1)*2, r_off:r_off+7] = W_s[2, :]
        W[(k_start+1)*2+1, r_off:r_off+7] = W_s[3, :]
        
        return 7
    
    # Helper to add a 2x2x1 Trivial block
    def add_trivial_2x2x1(U, V, W, r_off, k_start):
        U_t, V_t, W_t = _make_trivial_2x2x1_factors()
        # U_t is (4, 4). V_t is (2, 4). W_t is (2, 4).
        
        U[:, r_off:r_off+4] = U_t
        
        # V mapping:
        # Local b=0 (j=0, k_loc=0) -> Global b = 0*7 + k_start = k_start
        # Local b=1 (j=1, k_loc=0) -> Global b = 1*7 + k_start = 7 + k_start
        
        V[k_start, r_off:r_off+4] = V_t[0, :]
        V[7+k_start, r_off:r_off+4] = V_t[1, :]
        
        # W mapping:
        # Local c=0 (k_loc=0, i=0) -> Global c = k_start*2 + 0
        # Local c=1 (k_loc=0, i=1) -> Global c = k_start*2 + 1
        
        W[k_start*2, r_off:r_off+4] = W_t[0, :]
        W[k_start*2+1, r_off:r_off+4] = W_t[1, :]
        
        return 4
    
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 0)
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 2)
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 4)
    r_offset += add_trivial_2x2x1(U_sub, V_sub, W_sub, r_offset, 6)
    
    # Now we have a rank-25 decomposition for 2x2x7.
    # We need two such blocks for the full 2x4x7 problem.
    # Block 1: A[0:2, 0:2] (cols 0,1) * B[0:2, :] (rows 0,1)
    # Block 2: A[0:2, 2:4] (cols 2,3) * B[2:4, :] (rows 2,3)
    
    # Full dimensions:
    # n=2, m=4, p=7.
    # U_full: (8, 50)
    # V_full: (28, 50)
    # W_full: (14, 50)
    
    U_full = np.zeros((8, 50))
    V_full = np.zeros((28, 50))
    W_full = np.zeros((14, 50))
    
    # Block 1 (r_offset 0-24):
    # A indices: 0..3 (i=0,1; j=0,1). These map to U_full rows 0..3.
    # B indices: j in 0,1; k in 0..6. b = j*7 + k.
    #   j=0: b=0..6
    #   j=1: b=7..13
    # C indices: k in 0..6; i in 0,1. c = k*2 + i.
    #   c=0..13.
    
    U_full[0:4, 0:25] = U_sub
    # V_sub rows 0..13 correspond to b=0..13.
    V_full[0:14, 0:25] = V_sub
    # W_sub rows 0..13 correspond to c=0..13.
    W_full[0:14, 0:25] = W_sub
    
    # Block 2 (r_offset 25-49):
    # A indices: 4..7 (i=0,1; j=2,3).
    #   j=2: a=0*4+2=2? No. a = i*m + j. m=4.
    #   i=0, j=2 -> a=2.
    #   i=0, j=3 -> a=3.
    #   i=1, j=2 -> a=6.
    #   i=1, j=3 -> a=7.
    #   Wait, U_sub indices 0..3 correspond to local i,j.
    #   Local i=0, j=0 -> U_sub[0]. Global i=0, j=2 -> U_full[2].
    #   Local i=0, j=1 -> U_sub[1]. Global i=0, j=3 -> U_full[3].
    #   Local i=1, j=0 -> U_sub[2]. Global i=1, j=2 -> U_full[6].
    #   Local i=1, j=1 -> U_sub[3]. Global i=1, j=3 -> U_full[7].
    
    U_full[2, 25:50] = U_sub[0, :]
    U_full[3, 25:50] = U_sub[1, :]
    U_full[6, 25:50] = U_sub[2, :]
    U_full[7, 25:50] = U_sub[3, :]
    
    # B indices for Block 2:
    # B rows 2,3. j in 2,3.
    # b = j*7 + k.
    # j=2: b=14..20.
    # j=3: b=21..27.
    # V_sub rows 0..6 (local j=0) map to global j=2.
    # V_sub rows 7..13 (local j=1) map to global j=3.
    
    V_full[14:21, 25:50] = V_sub[0:7, :]
    V_full[21:28, 25:50] = V_sub[7:14, :]
    
    # C indices are the same for both blocks (addition).
    W_full[0:14, 25:50] = W_sub
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to trivial
        U_triv, V_triv, W_triv = _build_trivial(n, m, p)
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
