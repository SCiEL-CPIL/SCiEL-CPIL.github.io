"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7)
    """
    # u[i,j,m], v[j,k,m], w[k,i,m]
    # Indices: i,j,k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1: (A00+A11)(B00+B11)
    # u: A00(0), A11(3)
    U[0, 0] = 1; U[3, 0] = 1
    # v: B00(0), B11(3)
    V[0, 0] = 1; V[3, 0] = 1
    # w: C00(0)=1, C11(3)=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (A10+A11)B00
    # u: A10(2), A11(3)
    U[2, 1] = 1; U[3, 1] = 1
    # v: B00(0)
    V[0, 1] = 1
    # w: C10(2)=1, C11(3)=-1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3: A00(B01-B11)
    # u: A00(0)
    U[0, 2] = 1
    # v: B01(1), B11(3)
    V[1, 2] = 1; V[3, 2] = -1
    # w: C01(1)=1, C11(3)=1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4: A11(B10-B00)
    # u: A11(3)
    U[3, 3] = 1
    # v: B10(2), B00(0)
    V[2, 3] = 1; V[0, 3] = -1
    # w: C00(0)=1, C10(2)=1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5: (A00+A01)B11
    # u: A00(0), A01(1)
    U[0, 4] = 1; U[1, 4] = 1
    # v: B11(3)
    V[3, 4] = 1
    # w: C00(0)=-1, C01(1)=1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    # u: A10(2), A00(0)
    U[2, 5] = 1; U[0, 5] = -1
    # v: B00(0), B01(1)
    V[0, 5] = 1; V[1, 5] = 1
    # w: C11(3)=1
    W[3, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    # u: A01(1), A11(3)
    U[1, 6] = 1; U[3, 6] = -1
    # v: B10(2), B11(3)
    V[2, 6] = 1; V[3, 6] = 1
    # w: C00(0)=1
    W[0, 6] = 1
    
    return U, V, W

def _embed(U_local, V_local, W_local, offset_i, offset_j, offset_k, n, m, p, rank_total):
    """
    Embeds local decomposition into global matrices.
    U_local shape: (n_l*m_l, r_l)
    V_local shape: (m_l*p_l, r_l)
    W_local shape: (p_l*n_l, r_l)
    """
    n_l = U_local.shape[0] // m_l if m_l > 0 else 1 # This is tricky, better pass dimensions
    # Actually, let's pass dimensions explicitly to avoid confusion
    
def _construct_decomposition(n, m, p):
    """
    Recursively constructs a rank decomposition for n x m x p matrix multiplication.
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W, 1
    
    if n == 1:
        # Rank 1 decomposition for 1 x m x p
        # U (m, 1) all 1s
        # V (m*p, 1) all 1s
        # W (p, 1) all 1s
        U = [[1] for _ in range(m)]
        V = [[1] for _ in range(m*p)]
        W = [[1] for _ in range(p)]
        return U, V, W, 1
        
    if m == 1:
        # Rank 1 decomposition for n x 1 x p
        U = [[1] for _ in range(n)]
        V = [[1] for _ in range(p)]
        W = [[1] for _ in range(p*n)]
        return U, V, W, 1
        
    if p == 1:
        # Rank 1 decomposition for n x m x 1
        U = [[1] for _ in range(n*m)]
        V = [[1] for _ in range(m)]
        W = [[1] for _ in range(n)]
        return U, V, W, 1

    if n == 2 and m == 2 and p == 2:
        U, V, W = _get_strassen_2x2x2()
        return U.tolist(), V.tolist(), W.tolist(), 7

    # Recursive step
    # Split the largest dimension to balance the tree
    # Prioritize splitting n, then m, then p?
    # Or just split the first one > 1?
    # For 3,4,6:
    # Split n=3 -> 2, 1
    # R(2,4,6) + R(1,4,6)
    
    U_list = []
    V_list = []
    W_list = []
    rank = 0
    
    # Helper to add a block
    def add_block(U_b, V_b, W_b, r_b, off_i, off_j, off_k, n_b, m_b, p_b):
        # U_b is (n_b*m_b, r_b)
        # Global U is (n*m, R)
        # We need to append to U_list, etc.
        # But U_list is currently a list of rows? No, let's build arrays.
        pass

    # Let's use numpy arrays for accumulation
    U_acc = None
    V_acc = None
    W_acc = None
    current_rank = 0
    
    def merge(U_new, V_new, W_new, r_new):
        nonlocal U_acc, V_acc, W_acc, current_rank
        if U_acc is None:
            U_acc = np.array(U_new, dtype=float)
            V_acc = np.array(V_new, dtype=float)
            W_acc = np.array(W_new, dtype=float)
            current_rank = r_new
        else:
            U_acc = np.hstack([U_acc, np.array(U_new, dtype=float)])
            V_acc = np.hstack([V_acc, np.array(V_new, dtype=float)])
            W_acc = np.hstack([W_acc, np.array(W_new, dtype=float)])
            current_rank += r_new

    def embed_and_merge(U_local, V_local, W_local, r_local, off_i, off_j, off_k, n_l, m_l, p_l):
        # Create zero-padded global shapes for this block
        U_g = np.zeros((n * m, r_local))
        V_g = np.zeros((m * p, r_local))
        W_g = np.zeros((p * n, r_local))
        
        U_l = np.array(U_local, dtype=float)
        V_l = np.array(V_local, dtype=float)
        W_l = np.array(W_local, dtype=float)
        
        # Fill U
        for i in range(n_l):
            for j in range(m_l):
                row_g = (off_i + i) * m + (off_j + j)
                row_l = i * m_l + j
                U_g[row_g, :] = U_l[row_l, :]
                
        # Fill V
        for j in range(m_l):
            for k in range(p_l):
                row_g = (off_j + j) * p + (off_k + k)
                row_l = j * p_l + k
                V_g[row_g, :] = V_l[row_l, :]
                
        # Fill W
        for k in range(p_l):
            for i in range(n_l):
                row_g = (off_k + k) * n + (off_i + i)
                row_l = k * n_l + i
                W_g[row_g, :] = W_l[row_l, :]
                
        merge(U_g, V_g, W_g, r_local)

    # Determine split
    if n > 1:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1, r1 = _construct_decomposition(n1, m, p)
        embed_and_merge(U1, V1, W1, r1, 0, 0, 0, n1, m, p)
        
        U2, V2, W2, r2 = _construct_decomposition(n2, m, p)
        embed_and_merge(U2, V2, W2, r2, n1, 0, 0, n2, m, p)
        
    elif m > 1:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1, r1 = _construct_decomposition(n, m1, p)
        embed_and_merge(U1, V1, W1, r1, 0, 0, 0, n, m1, p)
        
        U2, V2, W2, r2 = _construct_decomposition(n, m2, p)
        embed_and_merge(U2, V2, W2, r2, 0, m1, 0, n, m2, p)
        
    else: # p > 1
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1, r1 = _construct_decomposition(n, m, p1)
        embed_and_merge(U1, V1, W1, r1, 0, 0, 0, n, m, p1)
        
        U2, V2, W2, r2 = _construct_decomposition(n, m, p2)
        embed_and_merge(U2, V2, W2, r2, 0, 0, p1, n, m, p2)

    return U_acc.tolist(), V_acc.tolist(), W_acc.tolist(), current_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Construct the decomposition
    U, V, W, rank = _construct_decomposition(n, m, p)
    
    # Verify
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))
    
    if not np.allclose(result, target, atol=1e-9):
        # Fallback to schoolbook if construction failed (should not happen)
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
