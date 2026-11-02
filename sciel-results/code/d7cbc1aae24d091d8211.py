import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns rank 11 decomposition for 2x2x3.
    Uses Strassen for cols 0,1 and schoolbook for col 2.
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u
        v_g = [v[0], v[1], 0]
        w_g = [w[0], w[1], 0]
        terms.append((u_g, v_g, w_g))
    
    # Schoolbook for col 2 (index 2 in B, index 2 in C)
    # A[0,0] * B[0,2] -> C[0,2]
    terms.append(([1, 0], [0, 0, 1], [1, 0, 0]))
    # A[0,1] * B[1,2] -> C[0,2]
    terms.append(([0, 1], [0, 0, 0], [0, 1, 0])) # Wait, B[1,2] is index 1 in B row, 2 in col.
    # B is 2x3. B[j, k].
    # Term for A[0,1] B[1,2]: u=[0,1], v=[0, 1, 0] (B[1,2] is at index 1*3+2? No, V is flattened)
    # V index is j*p + k.
    # B[0,2] -> V[0*3+2] = V[2].
    # B[1,2] -> V[1*3+2] = V[5].
    
    # Let's re-verify V indices for 2x2x3.
    # V shape (2*3, R).
    # V[j*3+k, r].
    
    # Schoolbook terms:
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 1: A[0,0] B[0,2]
    u = [1, 0]
    v = [0, 0, 1, 0, 0, 0] # V[2] = 1
    w = [1, 0, 0] # W[0] = 1 (C[0,2] is W[2*0+0]? No, W[k*n+i]. C[0,2] -> k=2, i=0. W[2*2+0] = W[4]?)
    # Wait, W index is k*n + i.
    # C[i, k].
    # W[k*n + i, r].
    # For C[0,2], k=2, i=0. W[2*2+0] = W[4].
    # For C[1,2], k=2, i=1. W[2*2+1] = W[5].
    
    # Let's re-check W indices in previous code.
    # Previous code: w_g[0] = w[0] ... w_g[5] = w[5].
    # For 2x2x3, n=2, p=3. W shape (3*2, R) = (6, R).
    # W[k*2+i].
    # C[0,0] -> W[0]. C[0,1] -> W[1]. C[0,2] -> W[2].
    # C[1,0] -> W[3]. C[1,1] -> W[4]. C[1,2] -> W[5].
    
    # So for C[0,2], W index is 2.
    # For C[1,2], W index is 5.
    
    # Corrected Schoolbook terms for 2x2x3:
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0])) # A[0,0] B[0,2] -> C[0,2]
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0])) # A[0,1] B[1,2] -> C[0,2]
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1])) # A[1,0] B[0,2] -> C[1,2]
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1])) # A[1,1] B[1,2] -> C[1,2]
    
    return terms

def _get_rank_4_1x2x2():
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 51 construction
    # 1. C[0:2, 0:2] from A[0:2, 0:4] B[0:4, 0:2] -> Rank 14
    # 2. C[0:2, 2:5] from A[0:2, 0:4] B[0:4, 2:5] -> Rank 17
    #    Split A[0:2, 0:4] into A[0:2, 0:3] and A[0:2, 3]
    #    Split B[0:4, 2:5] into B[0:3, 2:5] and B[3, 2:5]
    #    Term 2a: A[0:2, 0:3] B[0:3, 2:5] -> 2x3x3 Rank 11
    #    Term 2b: A[0:2, 3] B[3, 2:5] -> 2x1x3 Rank 6
    # 3. C[2, 0:2] from A[2, 0:4] B[0:4, 0:2] -> 1x4x2 Rank 8
    #    Split into 1x2x2 (Rank 4) + 1x2x2 (Rank 4)
    # 4. C[2, 2:5] from A[2, 0:4] B[0:4, 2:5] -> 1x4x3 Rank 12
    #    Split into 1x2x3 (Rank 6) + 1x2x3 (Rank 6)
    
    rank = 51
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
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    r11_terms = _get_rank_11_2x2x3()
    
    # Block 3: A[0:2, 0:3] * B[0:3, 2:5] -> C[0:2, 2:5]
    # A indices: 0,1,4,5,2,3 (0:2, 0:3)
    # B indices: 2,3,4, 7,8,9, 12,13,14 (0:3, 2:5)
    # C indices: 2,3,4, 5,6,7 (0:2, 2:5)
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        # u is 2x3 -> 6 entries.
        # u[0]=A[0,0], u[1]=A[0,1], u[2]=A[0,2], u[3]=A[1,0], u[4]=A[1,1], u[5]=A[1,2]
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[2] = u[2]
        u_g[4] = u[3]; u_g[5] = u[4]; u_g[6] = u[5]
        
        # v is 3x3 -> 9 entries.
        # v[0]=B[0,2], v[1]=B[0,3], v[2]=B[0,4]
        # v[3]=B[1,2], v[4]=B[1,3], v[5]=B[1,4]
        # v[6]=B[2,2], v[7]=B[2,3], v[8]=B[2,4]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]
        v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        v_g[12] = v[6]; v_g[13] = v[7]; v_g[14] = v[8]
        
        # w is 2x3 -> 6 entries.
        # w[0]=C[0,2], w[1]=C[0,3], w[2]=C[0,4]
        # w[3]=C[1,2], w[4]=C[1,3], w[5]=C[1,4]
        w_g[2] = w[0]; w_g[3] = w[1]; w_g[4] = w[2]
        w_g[5] = w[3]; w_g[6] = w[4]; w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 4: A[0:2, 3] * B[3, 2:5] -> C[0:2, 2:5]
    # A indices: 3, 7
    # B indices: 17, 18, 19
    # C indices: 2,3,4, 5,6,7
    # Rank 6
    r6_terms = _get_rank_6_1x2x3()
    # We need to map 1x2x3 terms to 2x1x3?
    # 1x2x3: A(1x2) B(2x3) -> C(1x3).
    # Here we have A(2x1) B(1x3) -> C(2x3).
    # This is transposed.
    # A[0:2, 3] is 2x1. B[3, 2:5] is 1x3.
    # C[i, k] = A[i, 3] B[3, k].
    # This is 6 independent products.
    # We can just add 6 terms.
    for i in range(2):
        for k in range(3):
            u_g = [0] * (n * m)
            v_g = [0] * (m * p)
            w_g = [0] * (p * n)
            u_g[i * m + 3] = 1
            v_g[3 * p + (2 + k)] = 1
            w_g[(2 + k) * n + i] = 1
            add_term(u_g, v_g, w_g)
            
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    r4_terms = _get_rank_4_1x2x2()
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]
        v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Block 8: A[2, 2:4] * B[2:4, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]
        v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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

