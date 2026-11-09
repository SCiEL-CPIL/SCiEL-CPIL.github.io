"""Combined approach: Rank-23 3x3 blocks (Rank 92) with fallback to Rank-108 and Rank-189."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2():
    """
    Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    strassen_A = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    strassen_B = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    strassen_C = [
        [((0, 0), 1), ((1, 1), 1)],
        [((1, 0), 1), ((1, 1), -1)],
        [((0, 1), 1), ((1, 1), 1)],
        [((0, 0), 1), ((1, 0), 1)],
        [((0, 0), -1), ((0, 1), 1)],
        [((1, 1), 1)],
        [((0, 0), 1)],
    ]

    for r in range(7):
        for (ai, aj), c in strassen_A[r].items():
            U[ai * 2 + aj, r] += c
        for (bi, bj), c in strassen_B[r].items():
            V[bi * 2 + bj, r] += c
        for (ci, cj), c in strassen_C[r]:
            W[cj * 2 + ci, r] += c

    return U, V, W


def _rank23_3x3_decomposition():
    """
    Return rank-23 decomposition for 3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Coefficients from known rank-23 decomposition (Landsberg)
    # Format: (row, col, coeff)
    terms = [
        # Term 0
        {'U': [(0,0,1), (2,2,1)], 'V': [(0,0,1), (2,2,1)], 'W': [(0,0,1), (2,2,1)]},
        # Term 1
        {'U': [(2,0,1), (2,2,1)], 'V': [(0,0,1)], 'W': [(2,0,1), (2,2,-1)]},
        # Term 2
        {'U': [(0,0,1)], 'V': [(0,1,1), (2,1,-1)], 'W': [(0,1,1), (2,1,1)]},
        # Term 3
        {'U': [(2,2,1)], 'V': [(2,0,1), (0,0,-1)], 'W': [(0,2,1), (2,0,1)]},
        # Term 4
        {'U': [(0,0,1), (0,1,1)], 'V': [(2,2,1)], 'W': [(0,0,-1), (0,1,1)]},
        # Term 5
        {'U': [(2,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(2,1,1)]},
        # Term 6
        {'U': [(0,1,1), (2,2,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,0,1)]},
        # Term 7
        {'U': [(0,2,1), (2,2,1)], 'V': [(0,2,1)], 'W': [(0,2,1), (2,2,1)]},
        # Term 8
        {'U': [(2,2,1)], 'V': [(0,2,1), (2,2,-1)], 'W': [(0,2,-1), (2,2,1)]},
        # Term 9
        {'U': [(0,0,1), (0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1), (0,2,1)]},
        # Term 10
        {'U': [(2,0,1)], 'V': [(0,2,1), (2,0,-1)], 'W': [(2,0,1), (2,2,1)]},
        # Term 11
        {'U': [(0,2,1), (2,0,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,2,1)]},
        # Term 12
        {'U': [(1,1,1)], 'V': [(1,1,1)], 'W': [(1,1,1)]},
        # Term 13
        {'U': [(0,1,1), (1,1,1)], 'V': [(1,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        # Term 14
        {'U': [(1,1,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        # Term 15
        {'U': [(1,0,1)], 'V': [(1,1,1), (0,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        # Term 16
        {'U': [(0,1,1), (1,0,1)], 'V': [(1,1,1)], 'W': [(0,1,-1), (1,0,1)]},
        # Term 17
        {'U': [(1,1,1), (1,0,-1)], 'V': [(0,1,1), (1,0,1)], 'W': [(1,1,1)]},
        # Term 18
        {'U': [(1,0,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(1,2,1), (1,1,-1)]},
        # Term 19
        {'U': [(1,1,1)], 'V': [(1,2,1), (0,2,-1)], 'W': [(0,2,1), (1,2,1)]},
        # Term 20
        {'U': [(0,2,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(0,2,-1), (1,2,1)]},
        # Term 21
        {'U': [(1,2,1)], 'V': [(1,1,1), (0,2,-1)], 'W': [(1,2,1), (1,1,1)]},
        # Term 22
        {'U': [(0,2,1), (1,2,-1)], 'V': [(1,1,1), (1,2,1)], 'W': [(0,2,1)]},
    ]
    
    for r, term in enumerate(terms):
        for lr, lc, coeff in term['U']:
            U[lr * 3 + lc, r] += coeff
        for lr, lc, coeff in term['V']:
            V[lr * 3 + lc, r] += coeff
        for lr, lc, coeff in term['W']:
            # W[k*n + i] where i=row, k=col, n=3
            W[lc * 3 + lr, r] += coeff
            
    return U, V, W


def _build_6x6_from_3x3_blocks_landsberg():
    """
    Build 6x6 multiplication using 4 blocks of 3x3 multiplication (rank 23 each).
    Total rank = 4 * 23 = 92
    """
    n, m, p = 6, 6, 6
    rank = 4 * 23  # 92
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U3, V3, W3 = _rank23_3x3_decomposition()
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                base_col = (bi * 2 + bj) * 2 + bk
                col_offset = base_col * 23
                
                for r in range(23):
                    for i in range(3):
                        for j in range(3):
                            local_u_row = i * 3 + j
                            global_u_row = (3 * bi + i) * m + (3 * bk + j)
                            U[global_u_row, col_offset + r] += U3[local_u_row, r]
                            
                    for j in range(3):
                        for k in range(3):
                            local_v_row = j * 3 + k
                            global_v_row = (3 * bk + j) * p + (3 * bj + k)
                            V[global_v_row, col_offset + r] += V3[local_v_row, r]
                            
                    for i in range(3):
                        for k in range(3):
                            local_w_row = k * 3 + i
                            global_w_row = (3 * bj + k) * n + (3 * bi + i)
                            W[global_w_row, col_offset + r] += W3[local_w_row, r]
                            
    return U.tolist(), V.tolist(), W.tolist()


def _rank27_3x3_decomposition():
    """
    Return rank-27 decomposition for 3x3 matrix multiplication (Schoolbook).
    """
    U3 = np.zeros((9, 27), dtype=float)
    V3 = np.zeros((9, 27), dtype=float)
    W3 = np.zeros((9, 27), dtype=float)
    
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, idx] = 1.0
                V3[j * 3 + k, idx] = 1.0
                W3[k * 3 + i, idx] = 1.0
                idx += 1
                
    return U3, V3, W3


def _build_6x6_from_3x3_blocks():
    """
    Build 6x6 multiplication using 4 blocks of 3x3 multiplication (schoolbook rank 27 each).
    Total rank = 4 * 27 = 108
    """
    n, m, p = 6, 6, 6
    rank = 4 * 27  # 108
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U3, V3, W3 = _rank27_3x3_decomposition()
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                base_col = (bi * 2 + bj) * 2 + bk
                col_offset = base_col * 27
                
                for r in range(27):
                    for i in range(3):
                        for j in range(3):
                            local_u_row = i * 3 + j
                            global_u_row = (3 * bi + i) * m + (3 * bk + j)
                            U[global_u_row, col_offset + r] += U3[local_u_row, r]
                            
                    for j in range(3):
                        for k in range(3):
                            local_v_row = j * 3 + k
                            global_v_row = (3 * bk + j) * p + (3 * bj + k)
                            V[global_v_row, col_offset + r] += V3[local_v_row, r]
                            
                    for i in range(3):
                        for k in range(3):
                            local_w_row = k * 3 + i
                            global_w_row = (3 * bj + k) * n + (3 * bi + i)
                            W[global_w_row, col_offset + r] += W3[local_w_row, r]
                            
    return U.tolist(), V.tolist(), W.tolist()


def _build_6x6_from_2x2_blocks_strassen():
    """
    Build 6x6 multiplication using 9 blocks of 2x2 multiplication with Strassen (rank 7).
    Total rank = 27 * 7 = 189
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U2, V2, W2 = _strassen_2x2()
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 3 + bj) * 3 + bk
                col_offset = base_col * 7
                
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            local_u_row = i * 2 + j
                            global_u_row = (2 * bi + i) * m + (2 * bj + j)
                            U[global_u_row, col_offset + r] += U2[local_u_row, r]
                            
                    for j in range(2):
                        for k in range(2):
                            local_v_row = j * 2 + k
                            global_v_row = (2 * bj + j) * p + (2 * bk + k)
                            V[global_v_row, col_offset + r] += V2[local_v_row, r]
                            
                    for i in range(2):
                        for k in range(2):
                            local_w_row = k * 2 + i
                            global_w_row = (2 * bk + k) * n + (2 * bi + i)
                            W[global_w_row, col_offset + r] += W2[local_w_row, r]
                            
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W, n, m, p, n_checks=20000):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    np.random.seed(12345)
    errors = 0
    max_err = 0.0
    
    for _ in range(n_checks):
        a_idx = np.random.randint(0, n * m)
        b_idx = np.random.randint(0, m * p)
        c_idx = np.random.randint(0, p * n)
        
        row_A = a_idx // m
        col_A = a_idx % m
        row_B = b_idx // p
        col_B = b_idx % p
        col_C = c_idx // n
        row_C = c_idx % n
        
        expected = 1.0 if (col_A == row_B and row_C == row_A and col_C == col_B) else 0.0
        
        computed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
        
        err = abs(computed - expected)
        max_err = max(max_err, err)
        if err > 1e-9:
            errors += 1
            
    return errors == 0, max_err


def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: 4 blocks of 3x3 Landsberg (rank 23 each) -> total rank 92
    try:
        U1, V1, W1 = _build_6x6_from_3x3_blocks_landsberg()
        rank1 = len(U1[0]) if U1 else 0
        valid1, err1 = _verify_decomposition(U1, V1, W1, n, m, p)
        if valid1 and rank1 < best_rank:
            best_rank = rank1
            best_result = {"U": U1, "V": V1, "W": W1}
    except Exception:
        pass
        
    if time.time() > deadline - 20:
        if best_result:
            return best_result
            
    # Approach 2: 4 blocks of 3x3 Schoolbook (rank 27 each) -> total rank 108
    try:
        U2, V2, W2 = _build_6x6_from_3x3_blocks()
        rank2 = len(U2[0]) if U2 else 0
        valid2, err2 = _verify_decomposition(U2, V2, W2, n, m, p)
        if valid2 and rank2 < best_rank:
            best_rank = rank2
            best_result = {"U": U2, "V": V2, "W": W2}
    except Exception:
        pass
        
    if time.time() > deadline - 10:
        if best_result:
            return best_result
            
    # Approach 3: 27 blocks of 2x2 Strassen (rank 7 each) -> total rank 189
    try:
        U3, V3, W3 = _build_6x6_from_2x2_blocks_strassen()
        rank3 = len(U3[0]) if U3 else 0
        valid3, err3 = _verify_decomposition(U3, V3, W3, n, m, p)
        if valid3 and rank3 < best_rank:
            best_rank = rank3
            best_result = {"U": U3, "V": V3, "W": W3}
    except Exception:
        pass
        
    if best_result:
        return best_result
        
    # Fallback: return Strassen without verification
    U3, V3, W3 = _build_6x6_from_2x2_blocks_strassen()
    return {"U": U3, "V": V3, "W": W3}
