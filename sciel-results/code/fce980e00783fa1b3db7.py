"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()
_DEADLINE = 130  # seconds, safety margin before 150s kill


def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification assuming numpy arrays already."""
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-7


def _compute_error(U, V, W, n, m, p):
    """Compute max absolute error of the decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))


def _build_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4x4 matrix multiplication using recursive Strassen.
    Returns U (16x49), V (16x49), W (16x49) as numpy arrays.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 patterns: (A_dict, B_dict, C_dict)
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # High-level terms as dicts {(block_row, block_col): coeff}
    high_level_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # For each hl term, which C blocks it contributes to: (c_block_row, c_block_col, coeff)
    hl_c_contrib = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(1, 0, 1), (1, 1, -1)],
        2: [(0, 1, 1), (1, 1, 1)],
        3: [(0, 0, 1), (1, 0, 1)],
        4: [(0, 0, -1), (0, 1, 1)],
        5: [(1, 1, 1)],
        6: [(0, 0, 1)],
    }
    
    col = 0
    for hl in range(7):
        A_hl_dict, B_hl_dict = high_level_terms[hl]
        
        for sub in range(7):
            A_sub_dict, B_sub_dict, C_sub_dict = strassen_2x2[sub]
            
            # U_term: 4x4, coefficient of A[i,j] for this term
            U_term = np.zeros((4, 4))
            for (br, bc), a_coeff in A_sub_dict.items():
                hl_coeff = A_hl_dict.get((br, bc), 0)
                total = a_coeff * hl_coeff
                if abs(total) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            U_term[r, c] += total
            
            # V_term: 4x4, coefficient of B[j,k] for this term
            V_term = np.zeros((4, 4))
            for (br, bc), b_coeff in B_sub_dict.items():
                hl_coeff = B_hl_dict.get((br, bc), 0)
                total = b_coeff * hl_coeff
                if abs(total) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            V_term[r, c] += total
            
            # W_term: 4x4, W[k,i] = coefficient for C[k,i]
            W_term = np.zeros((4, 4))
            for (cr, cc), c_sub in C_sub_dict.items():
                for (cbr, cbc, c_hl) in hl_c_contrib[hl]:
                    total = c_sub * c_hl
                    if abs(total) > 1e-10:
                        W_term[cbr*2+cr, cbc*2+cc] += total
            
            U[col*4:(col+1)*4, col] = U_term.flatten()
            V[col*4:(col+1)*4, col] = V_term.flatten()
            W[col*4:(col+1)*4, col] = W_term.flatten()
            col += 1
    
    return U, V, W


def _build_strassen_4x4x8():
    """
    Build a rank-98 decomposition for 4x4x8 matrix multiplication using two rank-49 4x4x4 decompositions.
    Returns U (16x98), V (32x98), W (32x98) as numpy arrays.
    """
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # First half: C1 = A * B1 (B1 = B[0:4, :])
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half: C2 = A * B2 (B2 = B[4:8, :])
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return U, V, W


def _quick_als(U, V, W, n, m, p, max_iter=50):
    """Quick ALS iterations."""
    rank = U.shape[1]
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                A = U[i*m + np.arange(m), :] * V[np.arange(m)*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to build a rank-98 decomposition using recursive Strassen
    try:
        U98, V98, W98 = _build_strassen_4x4x8()
        err98 = _compute_error(U98, V98, W98, n, m, p)
        if err98 < 1e-7:
            best_U, best_V, best_W = U98, V98, W98
            best_rank = 98
    except Exception as e:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
