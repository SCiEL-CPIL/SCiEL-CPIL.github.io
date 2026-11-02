"""Program for 2x3x9 matrix multiplication tensor decomposition using Strassen on 2x2 blocks."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s evaluator kill

def _check_time():
    return (time.time() - _START_TIME) > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    U: shape (4, 7) - A coefficients (flattened row-major: [a11, a12, a21, a22])
    V: shape (4, 7) - B coefficients (flattened row-major: [b11, b12, b21, b22])
    W: shape (4, 7) - C coefficients (flattened row-major: [c11, c12, c21, c22])
    
    Strassen:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    # U[i][r] = coefficient of A[i] in multiplication r
    # A flattened: index 0=a11, 1=a12, 2=a21, 3=a22
    # B flattened: index 0=b11, 1=b12, 2=b21, 3=b22
    # C flattened: index 0=c11, 1=c12, 2=c21, 3=c22
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c11 += M1, c22 += M1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c21 += M2, c22 -= M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c12 += M3, c22 += M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # c11 += M4, c21 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c11 -= M5, c12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c11 += M7
    
    return U, V, W

def construct():
    """
    Construct a rank-50 decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    1. Split A (2x3) into A_left (2x2) and A_right (2x1)
    2. Split B (3x9) into B_top (2x9) and B_bottom (1x9)
    3. A*B = A_left*B_top + A_right*B_bottom
    4. For A_left*B_top (2x2x9): split B_top into four 2x2 blocks + one 2x1 block
       - Each 2x2x2 uses Strassen (rank 7): 4 * 7 = 28
       - The 2x2x1 uses schoolbook (rank 4): 4
       - Subtotal: 32
    5. For A_right*B_bottom (2x1x9): schoolbook (rank 18)
    6. Total rank: 32 + 18 = 50
    
    Index conventions:
    - U has shape (n*m, rank) = (6, rank): A is 2x3, flattened row-major [a00,a01,a02,a10,a11,a12]
    - V has shape (m*p, rank) = (27, rank): B is 3x9, flattened row-major [b00..b08,b10..b18,b20..b28]
    - W has shape (p*n, rank) = (18, rank): C is 2x9, W[k*n+i] = C[i][k]
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # We'll build U, V, W as lists of lists (or numpy arrays)
    # Start with empty, append columns as we go
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a single term: u_val * A[u_idx] * v_val * B[v_idx] -> w_val * C[w_idx]"""
        U_cols.append({u_idx: u_val})
        V_cols.append({v_idx: v_val})
        W_cols.append({w_idx: w_val})
    
    def add_term_full(u_dict, v_dict, w_dict):
        """Add a term with possibly multiple indices."""
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)
    
    # A indexing: A[i][j] -> index i*3 + j (row-major, 2x3)
    # A[0][0]=0, A[0][1]=1, A[0][2]=2, A[1][0]=3, A[1][1]=4, A[1][2]=5
    # A_left = A[:, 0:2], A_right = A[:, 2]
    # A_left: [A[0][0], A[0][1], A[1][0], A[1][1]] = indices [0,1,3,4]
    # A_right: [A[0][2], A[1][2]] = indices [2,5]
    
    # B indexing: B[j][k] -> index j*9 + k (row-major, 3x9)
    # B_top = B[0:2, :], B_bottom = B[2, :]
    # B_top[j][k] -> index j*9 + k for j in {0,1}, k in 0..8
    # B_bottom[k] -> index 2*9 + k = 18 + k for k in 0..8
    
    # C indexing: C[i][k] -> W[k*n + i] = W[k*2 + i]
    # C[0][k] -> W[2k], C[1][k] -> W[2k+1]
    
    # Part 1: A_left * B_top (2x2 x 2x9)
    # A_left is 2x2: [A[0][0], A[0][1], A[1][0], A[1][1]]
    # B_top is 2x9: split into 4 (2x2) blocks + 1 (2x1) block
    # B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8], B_top[:, 8:9]
    
    su, sv, sw = _strassen_2x2x2()
    
    for block in range(4):
        k_start = block * 2
        k_end = k_start + 2
        if _check_time():
            break
        
        # Strassen block for columns k_start, k_start+1
        # A_left indices: a11=A[0][0]=0, a12=A[0][1]=1, a21=A[1][0]=3, a22=A[1][1]=4
        # B_top block: b11=B[0][k_start]=9*k_start, b12=B[0][k_start+1]=9*k_start+1
        #               b21=B[1][k_start]=9+9*k_start, b22=B[1][k_start+1]=9+9*k_start+1
        # C block: c11=C[0][k_start]=2*k_start, c12=C[0][k_start+1]=2*k_start+1
        #          c21=C[1][k_start]=2*k_start+2, c22=C[1][k_start+1]=2*k_start+3
        
        for r in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # U part: A_left coefficients
            if su[0, r] != 0: u_dict[0] = u_dict.get(0, 0) + su[0, r]  # A[0][0]
            if su[1, r] != 0: u_dict[1] = u_dict.get(1, 0) + su[1, r]  # A[0][1]
            if su[2, r] != 0: u_dict[3] = u_dict.get(3, 0) + su[2, r]  # A[1][0]
            if su[3, r] != 0: u_dict[4] = u_dict.get(4, 0) + su[3, r]  # A[1][1]
            
            # V part: B_top block coefficients
            if sv[0, r] != 0: v_dict[9*k_start] = v_dict.get(9*k_start, 0) + sv[0, r]       # B[0][k_start]
            if sv[1, r] != 0: v_dict[9*k_start+1] = v_dict.get(9*k_start+1, 0) + sv[1, r]   # B[0][k_start+1]
            if sv[2, r] != 0: v_dict[9+9*k_start] = v_dict.get(9+9*k_start, 0) + sv[2, r]   # B[1][k_start]
            if sv[3, r] != 0: v_dict[9+9*k_start+1] = v_dict.get(9+9*k_start+1, 0) + sv[3, r] # B[1][k_start+1]
            
            # W part: C block coefficients
            if sw[0, r] != 0: w_dict[2*k_start] = w_dict.get(2*k_start, 0) + sw[0, r]       # C[0][k_start]
            if sw[1, r] != 0: w_dict[2*k_start+1] = w_dict.get(2*k_start+1, 0) + sw[1, r]   # C[0][k_start+1]
            if sw[2, r] != 0: w_dict[2*k_start+2] = w_dict.get(2*k_start+2, 0) + sw[2, r]   # C[1][k_start]
            if sw[3, r] != 0: w_dict[2*k_start+3] = w_dict.get(2*k_start+3, 0) + sw[3, r]   # C[1][k_start+1]
            
            add_term_full(u_dict, v_dict, w_dict)
    
    # Last 2x2x1 block: B_top[:, 8] (just column 8)
    # A_left * B_top[:, 8]: C[0][8] += A[0][0]*B[0][8] + A[0][1]*B[1][8]
    #                       C[1][8] += A[1][0]*B[0][8] + A[1][1]*B[1][8]
    # Schoolbook: 4 terms
    # Term 1: A[0][0] * B[0][8] -> C[0][8]
    add_term_full({0: 1}, {8: 1}, {16: 1})  # W[8*2+0]=W[16]
    # Term 2: A[0][1] * B[1][8] -> C[0][8]
    add_term_full({1: 1}, {17: 1}, {16: 1})  # B[1][8]=9+8=17, W[16]
    # Term 3: A[1][0] * B[0][8] -> C[1][8]
    add_term_full({3: 1}, {8: 1}, {17: 1})  # W[8*2+1]=W[17]
    # Term 4: A[1][1] * B[1][8] -> C[1][8]
    add_term_full({4: 1}, {17: 1}, {17: 1})
    
    if _check_time():
        pass  # Continue to finish, we're close
    
    # Part 2: A_right * B_bottom (2x1 x 1x9)
    # A_right = [A[0][2], A[1][2]] = indices [2, 5]
    # B_bottom[k] = B[2][k] = index 18+k for k in 0..8
    # C[0][k] += A[0][2] * B[2][k] -> W[2k]
    # C[1][k] += A[1][2] * B[2][k] -> W[2k+1]
    # Schoolbook: 2*9 = 18 terms
    
    for k in range(9):
        if _check_time():
            break
        # A[0][2] * B[2][k] -> C[0][k]
        add_term_full({2: 1}, {18+k: 1}, {2*k: 1})
        # A[1][2] * B[2][k] -> C[1][k]
        add_term_full({5: 1}, {18+k: 1}, {2*k+1: 1})
    
    rank = len(U_cols)
    
    # Convert to matrix form
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx, r] = val
        for idx, val in V_cols[r].items():
            V[idx, r] = val
        for idx, val in W_cols[r].items():
            W[idx, r] = val
    
    # Verify correctness
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct from decomposition
    C_recon = np.zeros((n, p))
    for r in range(rank):
        a_vec = np.zeros(n*m)
        b_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        for idx, val in U_cols[r].items():
            a_vec[idx] = val
        for idx, val in V_cols[r].items():
            b_vec[idx] = val
        for idx, val in W_cols[r].items():
            w_vec[idx] = val
        
        a_flat = A_test.reshape(-1)
        b_flat = B_test.reshape(-1)
        prod = a_vec @ a_flat * (b_vec @ b_flat)
        c_flat = w_vec * prod
        C_recon += c_flat.reshape(p, n).T
    
    # Check
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        # Fallback to schoolbook rank 54
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        rank = n*m*p
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
