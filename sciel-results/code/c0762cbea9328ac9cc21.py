import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, num_tests=30, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition requires:
        sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1  for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
            return False
    return True


def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    
    return U, V, W


def _build_rank24_separate_terms(n, m, p):
    """
    Construct a rank-24 decomposition for 2x4x6 matrix multiplication.
    
    Strategy: For each (j, k) pair, create one rank-1 term that computes:
        A(0,j) * B(j,k) -> C(0,k)
        A(1,j) * B(j,k) -> C(1,k)
    
    Since n=2, m=4, p=6, there are m*p = 24 such pairs, giving rank 24.
    
    For each (j, k):
        - U[i*m + j, r] = 1 for i=0,1 (selects A(i,j))
        - V[j*p + k, r] = 1 (selects B(j,k))
        - W[k*n + i, r] = 1 for i=0,1 (routes to C(i,k))
    
    This is exact with integer entries.
    """
    rank = m * p  # 24
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U: select A(i,j) for all i
            for i in range(n):
                U[i*m + j, r] = 1.0
            # V: select B(j,k)
            V[j*p + k, r] = 1.0
            # W: route to C(i,k) for all i
            for i in range(n):
                W[k*n + i, r] = 1.0
    
    return U, V, W


def _build_block_strassen(n, m, p):
    """Build a block-strassen decomposition. Rank 42 for 2x4x6."""
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2 factors
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # M2 = (c+d)e
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    # M3 = a(f-h)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    # M4 = d(g-e)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    # M5 = (a+b)h
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    # M6 = (c-a)(e+g)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    # M7 = (b-d)(f+h)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    
    # W for Strassen: C00=M1+M4-M5+M7, C10=M2+M4, C01=M3+M5, C11=M1-M2+M3+M6
    # W index: k*2+i
    # row 0: C_{0,0} (k=0,i=0)
    # row 1: C_{1,0} (k=0,i=1)
    # row 2: C_{0,1} (k=1,i=0)
    # row 3: C_{1,1} (k=1,i=1)
    W_s[0, 0] = 1; W_s[0, 3] = 1; W_s[0, 4] = -1; W_s[0, 6] = 1
    W_s[1, 1] = 1; W_s[1, 3] = 1
    W_s[2, 2] = 1; W_s[2, 4] = 1
    W_s[3, 0] = 1; W_s[3, 1] = -1; W_s[3, 2] = 1; W_s[3, 5] = 1
    
    for a_block in range(2):
        for b_block in range(3):
            base_r = (a_block * 3 + b_block) * 7
            a_col_offset = a_block * 2
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            
            for r_s in range(7):
                r = base_r + r_s
                for i in range(2):
                    for j in range(2):
                        full_j = j + a_col_offset
                        U[i * m + full_j, r] = U_s[i * 2 + j, r_s]
                
                for j in range(2):
                    for k in range(2):
                        full_j = j + b_row_offset
                        full_k = k + b_col_offset
                        V[full_j * p + full_k, r] = V_s[j * 2 + k, r_s]
                
                for k in range(2):
                    for i in range(2):
                        full_k = k + b_col_offset
                        W[full_k * n + i, r] = W_s[k * 2 + i, r_s]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # First, try the rank-24 "separate terms" construction
    U24, V24, W24 = _build_rank24_separate_terms(n, m, p)
    
    if time.time() < deadline:
        is_valid = _verify_decomposition(U24, V24, W24, n, m, p, num_tests=30, seed=123)
        if is_valid:
            return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
    
    # Fallback to block Strassen (rank 42)
    U42, V42, W42 = _build_block_strassen(n, m, p)
    
    if time.time() < deadline:
        is_valid = _verify_decomposition(U42, V42, W42, n, m, p, num_tests=20, seed=123)
        if is_valid:
            return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Final fallback to trivial
    Ut, Vt, Wt = _build_trivial(n, m, p)
    return {"U": Ut.tolist(), "V": Vt.tolist(), "W": Wt.tolist()}

# EVOLVE-BLOCK-END
