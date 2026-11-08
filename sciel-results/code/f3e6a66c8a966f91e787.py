import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    rank = 60
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] += np.array(U_vec)
        V[:, col] += np.array(V_vec)
        W[:, col] += np.array(W_vec)
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    def add_strassen(a_base_i, b_base_i, c_base_i, k_base):
        """
        Strassen for A[a_base_i:a_base_i+2, 0:2] * B[b_base_i:b_base_i+2, k_base:k_base+2]
        -> C[c_base_i:c_base_i+2, k_base:k_base+2]
        """
        a00 = u_idx(a_base_i, 0)
        a01 = u_idx(a_base_i, 1)
        a10 = u_idx(a_base_i + 1, 0)
        a11 = u_idx(a_base_i + 1, 1)
        
        b00 = v_idx(b_base_i, k_base)
        b01 = v_idx(b_base_i, k_base + 1)
        b10 = v_idx(b_base_i + 1, k_base)
        b11 = v_idx(b_base_i + 1, k_base + 1)
        
        c00 = w_idx(c_base_i, k_base)
        c01 = w_idx(c_base_i, k_base + 1)
        c10 = w_idx(c_base_i + 1, k_base)
        c11 = w_idx(c_base_i + 1, k_base + 1)
        
        # m1: (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (a10+a11)*b00 -> c10 += m2, c11 -= m2
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a00*(b01-b11) -> c01 += m3, c11 += m3
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: a11*(b10-b00) -> c00 += m4, c10 += m4
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a00+a01)*b11 -> c00 -= m5, c01 += m5
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (a10-a00)*(b00+b01) -> c11 += m6
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (a01-a11)*(b10+b11) -> c00 += m7
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook_2x2x1(a_base_i, b_base_i, c_base_i, k_base):
        """Schoolbook for A[a_base_i:a_base_i+2, 0:2] * B[b_base_i:b_base_i+2, k_base:k_base+1] -> C[c_base_i:c_base_i+2, k_base:k_base+1]"""
        for i in range(2):
            for j in range(2):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(a_base_i + i, j)] = 1
                V_term[v_idx(b_base_i + j, k_base)] = 1
                W_term[w_idx(c_base_i + i, k_base)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # Part 1: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    # Split B columns into [0:2], [2:4], [4:6], [6:7]
    add_strassen(0, 0, 0, 0)   # 7 terms
    add_strassen(0, 0, 0, 2)   # 7 terms
    add_strassen(0, 0, 0, 4)   # 7 terms
    add_schoolbook_2x2x1(0, 0, 0, 6)  # 4 terms
    # Subtotal: 25 terms, col = 25

    # Part 2: A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]
    # M(1,2,7): rank 14
    for j in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(2, k)] = 1
            add_rank1_term(U_term, V_term, W_term)
    # Subtotal: 14 terms, col = 39

    # Part 3: A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]
    # M(3,1,7): rank 21
    for i in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(i, k)] = 1
            add_rank1_term(U_term, V_term, W_term)
    # Subtotal: 21 terms, col = 60

    if col != rank:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col2] = 1
                    V[j * p + k, col2] = 1
                    W[k * n + i, col2] = 1
                    col2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Quick verification with a single random test
    if time.time() + 5.0 < deadline - safety_margin:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_true = A_test @ B_test
        C_comp = np.zeros((n, p))
        
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_val = np.dot(U[i*m:(i+1)*m, r], A_test[i, :])
                    v_val = np.dot(V[k::p, r], B_test[:, k])
                    val += W[k*n+i, r] * u_val * v_val
                C_comp[i, k] = val
        
        if not np.allclose(C_comp, C_true, atol=1e-9):
            # Fallback
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col2] = 1
                        V[j * p + k, col2] = 1
                        W[k * n + i, col2] = 1
                        col2 += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
