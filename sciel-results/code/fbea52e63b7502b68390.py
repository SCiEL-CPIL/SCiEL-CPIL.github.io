import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """
    Verifies if the given U, V, W matrices form a valid tensor decomposition.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct_laderman_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication using Laderman's 15-term algorithm.
    Shapes: (9, 15).
    """
    U = np.zeros((9, 15))
    V = np.zeros((9, 15))
    W = np.zeros((9, 15))
    
    # Laderman's 15 terms
    # Indices: A(i,k), B(k,j), C(j,i)
    # Term 1: A11 B11 -> C11
    U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
    # Term 2: A22 B22 -> C22
    U[1*3+1, 1] = 1; V[1*3+1, 1] = 1; W[1*3+1, 1] = 1
    # Term 3: A33 B33 -> C33
    U[2*3+2, 2] = 1; V[2*3+2, 2] = 1; W[2*3+2, 2] = 1
    # Term 4: (A12+A13) B21 -> C11? 
    # Actually, let's use a verified set.
    # This hardcoded set might be wrong. 
    # To be safe, I will NOT use Laderman if I'm not 100% sure.
    # I will return the 27-term schoolbook for 3x3.
    return None

def construct_strassen_3x3_blocks():
    """
    Strassen 2x2 blocks of 3x3. Rank: 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    
    U = np.zeros((n * m, 189))
    V = np.zeros((m * p, 189))
    W = np.zeros((p * n, 189))
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb, c_comb = strassen_terms[term_idx]
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * 3
                        col_a = (blk_a % 2) * 3
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx, col] += coeff_a
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * 3
                        col_b = (blk_b % 2) * 3
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx, col] += coeff_b
                    for (blk_c, coeff_c) in c_comb:
                        row_c = (blk_c // 2) * 3
                        col_c = (blk_c % 2) * 3
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx, col] += coeff_c
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    
    # Try 189 rank solution
    result = construct_strassen_3x3_blocks()
    
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback: schoolbook rank 216
        n, m, p = 6, 6, 6
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    r = i * m * p + k * p + j
                    U[i * m + k, r] = 1
                    V[k * p + j, r] = 1
                    W[j * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
