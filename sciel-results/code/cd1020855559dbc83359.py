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


def _build_strassen_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen for C = A B where A, B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries: U[i*2+j, r] corresponds to coefficient of a_{i,j} in M_r
    # M1: (a+d)(e+h) -> a:1, d:1
    # M2: (c+d)e -> c:1, d:1
    # M3: a(f-h) -> a:1
    # M4: d(g-e) -> d:1
    # M5: (a+b)h -> a:1, b:1
    # M6: (c-a)(e+g) -> c:1, a:-1
    # M7: (b-d)(f+h) -> b:1, d:-1
    
    # V entries: V[j*2+k, r] corresponds to coefficient of b_{j,k} in M_r
    # M1: (a+d)(e+h) -> e:1, h:1
    # M2: (c+d)e -> e:1
    # M3: a(f-h) -> f:1, h:-1
    # M4: d(g-e) -> g:1, e:-1
    # M5: (a+b)h -> h:1
    # M6: (c-a)(e+g) -> e:1, g:1
    # M7: (b-d)(f+h) -> f:1, h:1
    
    # W entries: W[k*2+i, r] corresponds to coefficient in C_{i,k}
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: index = i*2+j
    # row 0: a (i=0,j=0)
    # row 1: b (i=0,j=1)
    # row 2: c (i=1,j=0)
    # row 3: d (i=1,j=1)
    
    U[0, 0] = 1; U[3, 0] = 1       # M1: a+d
    U[2, 1] = 1; U[3, 1] = 1       # M2: c+d
    U[0, 2] = 1                    # M3: a
    U[3, 3] = 1                    # M4: d
    U[0, 4] = 1; U[1, 4] = 1       # M5: a+b
    U[2, 5] = 1; U[0, 5] = -1      # M6: c-a
    U[1, 6] = 1; U[3, 6] = -1      # M7: b-d
    
    # V: index = j*2+k
    # row 0: e (j=0,k=0)
    # row 1: f (j=0,k=1)
    # row 2: g (j=1,k=0)
    # row 3: h (j=1,k=1)
    
    V[0, 0] = 1; V[3, 0] = 1       # M1: e+h
    V[0, 1] = 1                    # M2: e
    V[1, 2] = 1; V[3, 2] = -1      # M3: f-h
    V[2, 3] = 1; V[0, 3] = -1      # M4: g-e
    V[3, 4] = 1                    # M5: h
    V[0, 5] = 1; V[2, 5] = 1       # M6: e+g
    V[1, 6] = 1; V[3, 6] = 1       # M7: f+h
    
    # W: index = k*2+i
    # row 0: C_{0,0} (k=0,i=0)
    # row 1: C_{1,0} (k=0,i=1)
    # row 2: C_{0,1} (k=1,i=0)
    # row 3: C_{1,1} (k=1,i=1)
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1   # C00 = M1+M4-M5+M7
    W[1, 1] = 1; W[1, 3] = 1                              # C10 = M2+M4
    W[2, 2] = 1; W[2, 4] = 1                              # C01 = M3+M5
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1   # C11 = M1-M2+M3+M6
    
    return U, V, W


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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Build a rank-42 decomposition using block Strassen.
    # A is 2x4, B is 4x6.
    # Split A into two 2x2 blocks: A = [A1 | A2] where A1, A2 are 2x2
    # Split B into four 2x3 blocks: B = [[B11, B12], [B21, B22]] where each is 2x3
    # Actually, let's split B into 2x2 blocks. B is 4x6 = [2x2 | 2x2 | 2x2] in two rows.
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x2.
    # C = A1*B1 + A2*B2 where B1 = [B11, B12, B13] is 2x6, B2 = [B21, B22, B23] is 2x6.
    # So C = A1 * [B11,B12,B13] + A2 * [B21,B22,B23]
    # Each term A1*Bij is 2x2 * 2x2, which costs 7 via Strassen.
    # Total: 2 * 3 * 7 = 42.
    
    # Let's build this carefully.
    # U has shape (n*m, rank) = (2*4, rank) = (8, rank)
    # V has shape (m*p, rank) = (4*6, rank) = (24, rank)
    # W has shape (p*n, rank) = (6*2, rank) = (12, rank)
    
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2_2x2()
    
    # For each of the 2 A-blocks (a_block in {0,1}) and 3 B-blocks (b_block in {0,1,2}):
    # a_block=0: A rows 0-1, cols 0-1 (A1)
    # a_block=1: A rows 0-1, cols 2-3 (A2)
    # b_block=0: B rows 0-1, cols 0-1 (B11)
    # b_block=1: B rows 0-1, cols 2-3 (B12)
    # b_block=2: B rows 0-1, cols 4-5 (B13)
    # For a_block=0, we use B rows 0-1 (B11, B12, B13)
    # For a_block=1, we use B rows 2-3 (B21, B22, B23)
    
    for a_block in range(2):
        for b_block in range(3):
            base_r = (a_block * 3 + b_block) * 7
            a_col_offset = a_block * 2  # cols 0-1 or 2-3 of A
            b_row_offset = a_block * 2  # rows 0-1 or 2-3 of B
            b_col_offset = b_block * 2  # cols 0-1, 2-3, or 4-5 of B
            
            for r_s in range(7):
                r = base_r + r_s
                # U: A block entries
                for i in range(2):
                    for j in range(2):
                        full_j = j + a_col_offset
                        U[i * m + full_j, r] = U_s[i * 2 + j, r_s]
                
                # V: B block entries
                for j in range(2):
                    for k in range(2):
                        full_j = j + b_row_offset
                        full_k = k + b_col_offset
                        V[full_j * p + full_k, r] = V_s[j * 2 + k, r_s]
                
                # W: output entries
                for k in range(2):
                    for i in range(2):
                        full_k = k + b_col_offset
                        W[full_k * n + i, r] = W_s[k * 2 + i, r_s]
    
    # Verify
    if time.time() < deadline:
        is_valid = _verify_decomposition(U, V, W, n, m, p, num_tests=20, seed=123)
        if is_valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to trivial
    U_t, V_t, W_t = _build_trivial(n, m, p)
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

# EVOLVE-BLOCK-END
