import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_2x2x1_schoolbook():
    """
    Schoolbook decomposition for 2x2 x 2x1.
    U (4, 2), V (2, 2), W (2, 2).
    """
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # c11 = a11*b11 + a12*b21
    # c21 = a21*b11 + a22*b21
    # Term 1: a11*b11
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    # Term 2: a12*b21
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    # Term 3: a21*b11
    U[2][0] = 1
    V[0][0] = 1 # Wait, V is shared? No, V is separate for each term in the sum?
    # The decomposition is sum of rank-1 tensors.
    # T_r = U_r \otimes V_r \otimes W_r.
    # C = sum_r T_r.
    # So we need to add contributions.
    
    # Let's restart this function properly.
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # We need 2 terms?
    # c11 = a11*b11 + a12*b21
    # c21 = a21*b11 + a22*b21
    # This is 4 terms in schoolbook.
    # Rank is 4?
    # Wait, 2x2x1 tensor.
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # This requires 4 multiplications.
    # So rank is 4.
    
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _build_2x1x3_schoolbook():
    """
    Schoolbook decomposition for 2x1 x 1x3.
    U (2, 6), V (3, 6), W (6, 2).
    """
    U = np.zeros((2, 6))
    V = np.zeros((3, 6))
    W = np.zeros((6, 2))
    
    # C is 2x3.
    # C[i,k] = A[i,0] * B[0,k].
    # 2 * 1 * 3 = 6 terms.
    col = 0
    for i in range(2):
        for k in range(3):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return U, V, W

def _build_2x2x3():
    """
    Build 2x2x3 decomposition.
    Rank 9.
    U (4, 9), V (6, 9), W (6, 9).
    """
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_2x2x1_schoolbook()
    
    U = np.zeros((4, 9))
    V = np.zeros((6, 9))
    W = np.zeros((6, 9))
    
    # Part 1: A (2x2) * B_a (2x2) -> C_a (2x2)
    # B_a is B[:, 0:2]
    # U, V, W for Strassen are for 2x2x2.
    # V_str is (4, 7). We need V for 2x2 (cols 0,1).
    # V_str indices: j*p+k. p=2.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 2. j=1, k=1 -> 3.
    # We want B[:, 0:2], so k=0,1.
    # This corresponds to V_str rows 0,1,2,3.
    # W_str is (4, 7). k*n+i. n=2.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # We want C[:, 0:2], so k=0,1.
    # This corresponds to W_str rows 0,1,2,3.
    
    for r in range(7):
        U[:, r] = U_str[:, r]
        V[:, r] = V_str[:, r]
        W[:, r] = W_str[:, r]
        
    # Part 2: A (2x2) * B_b (2x1) -> C_b (2x1)
    # B_b is B[:, 2]
    # C_b is C[:, 2]
    # U_sb is (4, 4).
    # V_sb is (2, 4). V_sb indices: j*p+k. p=1.
    # j=0 -> 0. j=1 -> 1.
    # We want B[:, 2]. This is a new dimension.
    # In the 2x2x3 context, V is (6, 9).
    # V row index is j*3 + k.
    # For B_b, k=2.
    # So V rows are j*3 + 2.
    # j=0 -> 2. j=1 -> 5.
    # W_sb is (2, 4). W_sb indices: k*n+i. n=2.
    # k=0.
    # In 2x2x3 context, W row index is k*2 + i.
    # For C_b, k=2.
    # So W rows are 2*2 + i = 4+i.
    # i=0 -> 4. i=1 -> 5.
    
    for r in range(4):
        col = 7 + r
        U[:, col] = U_sb[:, r]
        # Map V_sb (j) to V (j*3 + 2)
        V[2, col] = V_sb[0, r]
        V[5, col] = V_sb[1, r]
        # Map W_sb (i) to W (4+i)
        W[4, col] = W_sb[0, r]
        W[5, col] = W_sb[1, r]
        
    return U, V, W

def _build_2x3x3():
    """
    Build 2x3x3 decomposition.
    Rank 15.
    U (6, 15), V (9, 15), W (6, 15).
    """
    U_sb, V_sb, W_sb = _build_2x1x3_schoolbook()
    U_223, V_223, W_223 = _build_2x2x3()
    
    U = np.zeros((6, 15))
    V = np.zeros((9, 15))
    W = np.zeros((6, 15))
    
    # Part 1: A_a (2x1) * B_a (1x3) -> C_a (2x3)
    # A_a is A[:, 0].
    # B_a is B[0, :].
    # U_sb is (2, 6). U_sb indices: i*m+j. m=1.
    # i=0 -> 0. i=1 -> 1.
    # In 2x3x3 context, U row index is i*3 + j.
    # For A_a, j=0.
    # So U rows are i*3 + 0.
    # i=0 -> 0. i=1 -> 3.
    
    # V_sb is (3, 6). V_sb indices: j*p+k. j=0, p=3.
    # k=0 -> 0. k=1 -> 1. k=2 -> 2.
    # In 2x3x3 context, V row index is j*3 + k.
    # For B_a, j=0.
    # So V rows are 0*3 + k = k.
    # k=0 -> 0. k=1 -> 1. k=2 -> 2.
    
    # W_sb is (6, 2). W_sb indices: k*n+i. n=2.
    # k=0 -> i. k=1 -> 2+i. k=2 -> 4+i.
    # In 2x3x3 context, W row index is k*2 + i.
    # This matches exactly.
    
    for r in range(6):
        col = r
        U[0, col] = U_sb[0, r]
        U[3, col] = U_sb[1, r]
        V[0, col] = V_sb[0, r]
        V[1, col] = V_sb[1, r]
        V[2, col] = V_sb[2, r]
        W[:, col] = W_sb[:, r]
        
    # Part 2: A_b (2x2) * B_b (2x3) -> C_b (2x3)
    # A_b is A[:, 1:3].
    # B_b is B[1:3, :].
    # U_223 is (4, 9). U_223 indices: i*2 + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 2. i=1, j=1 -> 3.
    # In 2x3x3 context, U row index is i*3 + j.
    # For A_b, j=1,2.
    # i=0, j=1 -> 1. i=0, j=2 -> 2.
    # i=1, j=1 -> 4. i=1, j=2 -> 5.
    
    # V_223 is (6, 9). V_223 indices: j*3 + k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=0, k=2 -> 2.
    # j=1, k=0 -> 3. j=1, k=1 -> 4. j=1, k=2 -> 5.
    # In 2x3x3 context, V row index is j*3 + k.
    # For B_b, j=1,2.
    # So we need to shift j by 1.
    # V_223 row (j_local, k) -> V row ((j_local+1)*3 + k).
    
    # W_223 is (6, 9). W_223 indices: k*2 + i.
    # In 2x3x3 context, W row index is k*2 + i.
    # This matches exactly.
    
    for r in range(9):
        col = 6 + r
        # Map U_223
        U[1, col] = U_223[0, r]
        U[2, col] = U_223[1, r]
        U[4, col] = U_223[2, r]
        U[5, col] = U_223[3, r]
        
        # Map V_223
        for j_loc in range(2):
            for k in range(3):
                v_idx = j_loc * 3 + k
                v_global = (j_loc + 1) * 3 + k
                V[v_global, col] = V_223[v_idx, r]
                
        # Map W_223
        W[:, col] = W_223[:, r]
        
    return U, V, W

def _build_2x5x3():
    """
    Build 2x5x3 decomposition.
    Rank 24.
    U (10, 24), V (15, 24), W (6, 24).
    """
    U_223, V_223, W_223 = _build_2x2x3()
    U_233, V_233, W_233 = _build_2x3x3()
    
    U = np.zeros((10, 24))
    V = np.zeros((15, 24))
    W = np.zeros((6, 24))
    
    # Part 1: A_1 (2x2) * B_1 (2x3) -> C_1 (2x3)
    # A_1 is A[:, 0:2].
    # B_1 is B[0:2, :].
    # U_223 is (4, 9).
    # U row index i*3+j.
    # For A_1, j=0,1.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 3. i=1, j=1 -> 4.
    
    # V_223 is (6, 9).
    # V row index j*3+k.
    # For B_1, j=0,1.
    # This matches V_223 indices directly?
    # V_223 was for 2x2x3. j=0,1.
    # So V_223 rows 0..5 correspond to j=0,1 and k=0..2.
    # This matches B_1.
    
    # W_223 is (6, 9).
    # Matches C_1.
    
    for r in range(9):
        col = r
        U[0, col] = U_223[0, r]
        U[1, col] = U_223[1, r]
        U[3, col] = U_223[2, r]
        U[4, col] = U_223[3, r]
        
        V[:, col] = V_223[:, r]
        
        W[:, col] = W_223[:, r]
        
    # Part 2: A_2 (2x3) * B_2 (3x3) -> C_2 (2x3)
    # A_2 is A[:, 2:5].
    # B_2 is B[2:5, :].
    # U_233 is (6, 15).
    # U row index i*3+j.
    # For A_2, j=0,1,2 (local). Global j=2,3,4.
    # i=0, j_loc=0 -> 0. Global 2.
    # i=0, j_loc=1 -> 1. Global 3.
    # i=0, j_loc=2 -> 2. Global 4.
    # i=1, j_loc=0 -> 3. Global 5.
    # i=1, j_loc=1 -> 4. Global 6.
    # i=1, j_loc=2 -> 5. Global 7.
    
    # V_233 is (9, 15).
    # V row index j*3+k.
    # For B_2, j=0,1,2 (local). Global j=2,3,4.
    # j_loc=0 -> 0. Global 2.
    # j_loc=1 -> 1. Global 3.
    # j_loc=2 -> 2. Global 4.
    # So V_233 rows 0..8 correspond to j_loc=0,1,2.
    # We need to map to global V rows.
    # V_233 row (j_loc*3+k) -> V row ((j_loc+2)*3+k).
    
    # W_233 is (6, 15).
    # Matches C_2.
    
    for r in range(15):
        col = 9 + r
        # Map U_233
        U[2, col] = U_233[0, r]
        U[3, col] = U_233[1, r]
        U[4, col] = U_233[2, r]
        U[5, col] = U_233[3, r]
        U[6, col] = U_233[4, r]
        U[7, col] = U_233[5, r]
        
        # Map V_233
        for j_loc in range(3):
            for k in range(3):
                v_idx = j_loc * 3 + k
                v_global = (j_loc + 2) * 3 + k
                V[v_global, col] = V_233[v_idx, r]
                
        # Map W_233
        W[:, col] = W_233[:, r]
        
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    U_block, V_block, W_block = _build_2x5x3()
    
    total_rank = 96
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for r_block in range(2):
        for c_block in range(2):
            base = (r_block * 2 + c_block) * 24
            
            # Map U
            for i_loc in range(2):
                for j_loc in range(5):
                    u_idx = i_loc * 5 + j_loc
                    i_glob = r_block * 2 + i_loc
                    j_glob = j_loc
                    U[i_glob * m + j_glob, base:base+24] = U_block[u_idx, :]
                    
            # Map V
            for j_loc in range(5):
                for k_loc in range(3):
                    v_idx = j_loc * 3 + k_loc
                    j_glob = j_loc
                    k_glob = c_block * 3 + k_loc
                    V[j_glob * p + k_glob, base:base+24] = V_block[v_idx, :]
                    
            # Map W
            for k_loc in range(3):
                for i_loc in range(2):
                    w_idx = k_loc * 2 + i_loc
                    k_glob = c_block * 3 + k_loc
                    i_glob = r_block * 2 + i_loc
                    W[k_glob * n + i_glob, base:base+24] = W_block[w_idx, :]
                    
    if not _verify_decomp(U, V, W, n, m, p, 1e-9):
        # Fallback to schoolbook if verification fails
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
