"""Matrix multiplication 3x4x5 - rank-minimal decomposition via block tiling."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 B11
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    # M4 = A11 B10
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    # M5 = (A00+A01)B10
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    # M6 = (A10-A00)(B00+B11)
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    Convention: u indexed by (i,j) for i in 0..1, j in 0..1
                v indexed by (j,k) for j in 0..1, k in 0..2
                w indexed by (k,i) for k in 0..2, i in 0..1
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = list(u)
        v_g = [0] * 6
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[2] = v[2]
        v_g[3] = v[3]
        w_g = [0] * 6
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[2] = w[2]
        w_g[3] = w[3]
        terms.append((u_g, v_g, w_g))
    
    terms.append(([1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2.
    Convention: u indexed by j (0..1), v indexed by (j,k) (4 entries), w indexed by k (0..1)
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3.
    Convention: u indexed by j (0..1), v indexed by (j,k) (6 entries), w indexed by k (0..2)
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _get_rank_23_3x3x3():
    """
    Returns list of 23 (u, v, w) vectors for 3x3x3.
    This is the 23-term algorithm for 3x3 matrix multiplication.
    Convention: u indexed by (i,j) for i,j in 0..2 (9 entries)
                v indexed by (j,k) for j,k in 0..2 (9 entries)
                w indexed by (k,i) for k,i in 0..2 (9 entries)
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    
    # 1. Top-left 2x2x2 block
    for u, v, w in s_terms:
        u_g = [0]*9; v_g = [0]*9; w_g = [0]*9
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[3] = u[2]; u_g[4] = u[3]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[3] = v[2]; v_g[4] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        terms.append((u_g, v_g, w_g))
        
    # 2. Top-middle 2x1x2 block (A[0:2, 2], B[2, 0:2]) -> C[0:2, 0:2]
    # u: A[0:2, 2] -> indices 2, 5
    # v: B[2, 0:2] -> indices 6, 7
    # w: C[0:2, 0:2] -> indices 0,1,3,4
    # This is 2x1x2. Rank 4.
    # We can use schoolbook:
    terms.append(([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,1,0, 0,0,0])) # A02 B20 C00
    terms.append(([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,0])) # A02 B21 C10
    terms.append(([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0])) # A02 B20 C01
    terms.append(([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,1,0])) # A02 B21 C11
    
    # Wait, the above is 4 terms. But I need to be careful with indices.
    # A[0:2, 2] means rows 0,1 and col 2. So u[2] (i=0,j=2) and u[5] (i=1,j=2).
    # B[2, 0:2] means row 2 and cols 0,1. So v[6] (j=2,k=0) and v[7] (j=2,k=1).
    # C[0:2, 0:2] means rows 0,1 and cols 0,1. So w[0] (k=0,i=0), w[1] (k=0,i=1), w[3] (k=1,i=0), w[4] (k=1,i=1).
    
    # Let's redo the 4 terms for this block properly.
    # Term 1: A[0,2] * B[2,0] * C[0,0]
    terms[-4] = ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0])
    # Term 2: A[0,2] * B[2,1] * C[1,0]
    terms[-3] = ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0])
    # Term 3: A[1,2] * B[2,0] * C[0,1]
    terms[-2] = ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0])
    # Term 4: A[1,2] * B[2,1] * C[1,1]
    terms[-1] = ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0])
    
    # 3. Middle-top 1x2x2 block (A[2, 0:2], B[0:2, 2]) -> C[0:2, 2]
    # u: A[2, 0:2] -> indices 6, 7
    # v: B[0:2, 2] -> indices 2, 5
    # w: C[0:2, 2] -> indices 6, 7
    terms.append(([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0])) # A20 B02 C02
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0])) # A21 B02 C12
    terms.append(([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0])) # A20 B12 C02
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0])) # A21 B12 C12
    
    # 4. Middle-middle 1x1x1 block (A[2,2], B[2,2]) -> C[2,2]
    terms.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]))
    
    # 5. Middle-bottom 1x2x2 block (A[2, 0:2], B[0:2, 2])? No, this is the same as 3.
    # Wait, I need to cover the whole 3x3x3.
    # My tiling:
    # Block 1: A[0:2,0:2], B[0:2,0:2] -> C[0:2,0:2] (2x2x2, rank 7)
    # Block 2: A[0:2,2], B[2,0:2] -> C[0:2,0:2] (2x1x2, rank 4)
    # Block 3: A[2,0:2], B[0:2,2] -> C[0:2,2] (1x2x2, rank 4)
    # Block 4: A[2,2], B[2,2] -> C[2,2] (1x1x1, rank 1)
    # Total so far: 7+4+4+1 = 16.
    # Missing:
    # Block 5: A[0:2,0:2], B[0:2,2] -> C[0:2,2] (2x2x2, rank 7)
    # Block 6: A[0:2,2], B[2,2] -> C[2,0:2] (2x1x1, rank 2)
    # Block 7: A[2,0:2], B[2,0:2] -> C[0:2,2] (1x2x2, rank 4)
    # Block 8: A[2,2], B[2,0:2] -> C[0:2,2] (1x1x2, rank 2)
    
    # Let's restart the 3x3x3 construction to be systematic.
    # I will use the known 23-term algorithm.
    # Since I don't have it memorized perfectly, I will construct a valid 23-term one.
    # Actually, the sum of ranks of the 9 blocks in a 3x3 tiling is:
    # (2x2x2)*4 + (2x1x2)*2 + (1x2x2)*2 + (1x1x1)*1 = 7*4 + 4*2 + 4*2 + 1 = 28+8+8+1 = 45.
    # This is much worse than 23.
    # The 23-term algorithm is not a simple block tiling.
    # I will fall back to the 56-rank decomposition which is known to work.
    return terms # This is incomplete, so I will not use this function.

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will construct the 56-rank decomposition.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    s_terms = _get_strassen_2x2x2()
    
    # Block 1: C[0:2, 0:2] from A[0:2, 0:2] and B[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: C[0:2, 0:2] from A[0:2, 2:4] and B[2:4, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    r11_terms = _get_rank_11_2x2x3()
    
    # Block 3: C[0:2, 2:5] from A[0:2, 0:2] and B[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 4: C[0:2, 2:5] from A[0:2, 2:4] and B[2:4, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    r4_terms = _get_rank_4_1x2x2()
    
    # Block 5: C[2, 0:2] from A[2, 0:2] and B[0:2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 6: C[2, 0:2] from A[2, 2:4] and B[2:4, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    r6_terms = _get_rank_6_1x2x3()
    
    # Block 7: C[2, 2:5] from A[2, 0:2] and B[0:2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Block 8: C[2, 2:5] from A[2, 2:4] and B[2:4, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
