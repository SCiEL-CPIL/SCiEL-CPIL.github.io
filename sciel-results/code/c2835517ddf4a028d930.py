import time
import numpy as np

TIME_LIMIT_SECONDS = 110
START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - START_TIME

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from U, V, W: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The matrix multiplication tensor should be: T[i,j,k] = 1 if i==k else 0 (for A[i,j]*B[j,k] -> C[i,k])
    # Actually, the convention is W[k*n+i, r], so W is indexed by (k, i)
    # The tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta(i, k) for the matrix multiplication tensor
    
    # Let's verify by checking a few random cases
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        # Compute C using the decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_vec = U[:, r]  # shape (n*m,)
            v_vec = V[:, r]  # shape (m*p,)
            w_vec = W[:, r]  # shape (p*n,)
            
            # Reshape u_vec to (n, m), v_vec to (m, p), w_vec to (p, n)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The contribution is: sum over j of A[i,j] * u_mat[i,j] * (B @ v_mat_col) ... 
            # Actually, the bilinear form is:
            # C[i,k] += sum_j A[i,j] * (sum_j' B[j',k] * V[j',k,r]) * U[i,j,r] * W[k*n+i,r]
            # Wait, let me reconsider. The tensor decomposition means:
            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) * W[k*n+i,r]
            
            a_factors = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_factors[i] += A[i,j] * U[i*m+j, r]
            
            b_factors = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_factors[k] += B[j,k] * V[j*p+k, r]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += a_factors[i] * b_factors[k] * W[k*n+i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def construct():
    n, m, p = 4, 4, 5
    
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
        
    # Strassen 4x4x4
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    s_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2+c
            elif b_idx == 2: gi, gj = 2+r, c
            elif b_idx == 3: gi, gj = 2+r, 2+c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2+c
            elif b_idx == 2: gj, gk = 2+r, c
            elif b_idx == 3: gj, gk = 2+r, 2+c
            res.append((gj, gk, coeff))
        return res

    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            p_elems, q_elems = inner_terms[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0: continue
                    
                    total_coeff = sc * cc
                    if total_coeff == 0: continue
                    
                    if c_pos == 0: gi, gk = s_r, s_c
                    elif c_pos == 1: gi, gk = s_r, 2+s_c
                    elif c_pos == 2: gi, gk = 2+s_r, s_c
                    elif c_pos == 3: gi, gk = 2+s_r, 2+s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)

    # Schoolbook 4x4x1
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)

    # Verify the decomposition is correct
    if _elapsed() < TIME_LIMIT_SECONDS - 5:
        if not _verify_decomposition(U, V, W, n, m, p, rank):
            # If verification fails, fall back to pure schoolbook (rank 80)
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col_fb] = 1
                        V_fb[j * p + k][col_fb] = 1
                        W_fb[k * n + i][col_fb] = 1
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
