import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best result before 140s
    def deadline_ok():
        return (time.time() - start_time) < 140
    
    # Primary: rank-88 Strassen-based decomposition
    try:
        result = _construct_rank88(n, m, p)
        if result is not None:
            # Quick verification
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    if not deadline_ok():
        # Fallback: schoolbook rank 96 (always valid)
        return _construct_schoolbook(n, m, p)
    
    return _construct_schoolbook(n, m, p)


def _construct_rank88(n, m, p):
    """
    Rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Strategy:
    - Row 0: schoolbook 1x4x8 -> rank 32
    - Rows 1-2: Strassen on 2x4x8 -> 8 blocks of 2x2x2, each rank 7 -> rank 56
    Total: 88
    """
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1
        
    # Part 1: Schoolbook for A[0,:] (1x4) * B (4x8) -> C[0,:] (1x8)
    # C[0,k] = sum_j A[0,j] * B[j,k]
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Strassen for A[1:3,:] (2x4) * B (4x8) -> C[1:3,:] (2x8)
    # A col blocks: [0,1] and [2,3]
    a_col_blocks = [(0, 1), (2, 3)]
    # B row blocks: [0,1] and [2,3]
    b_row_blocks = [(0, 1), (2, 3)]
    # B col blocks: [0,1], [2,3], [4,5], [6,7]
    b_col_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for a_cb in a_col_blocks:
        for b_rb in b_row_blocks:
            for b_cb in b_col_blocks:
                a_r1, a_r2 = 1, 2
                a_c1, a_c2 = a_cb
                b_r1, b_r2 = b_rb
                b_c1, b_c2 = b_cb
                c_r1, c_r2 = 1, 2
                c_c1, c_c2 = b_cb
                
                u_a = a_r1 * m + a_c1
                u_b = a_r1 * m + a_c2
                u_c = a_r2 * m + a_c1
                u_d = a_r2 * m + a_c2
                
                v_e = b_r1 * p + b_c1
                v_f = b_r1 * p + b_c2
                v_g = b_r2 * p + b_c1
                v_h = b_r2 * p + b_c2
                
                w_c11 = b_c1 * n + c_r1
                w_c12 = b_c2 * n + c_r1
                w_c21 = b_c1 * n + c_r2
                w_c22 = b_c2 * n + c_r2
                
                # M1 = (a+d)(e+h)
                add_rank_unit(
                    {u_a: 1, u_d: 1},
                    {v_e: 1, v_h: 1},
                    {w_c11: 1, w_c22: 1}
                )
                # M2 = (c+d)(e)
                add_rank_unit(
                    {u_c: 1, u_d: 1},
                    {v_e: 1},
                    {w_c21: 1, w_c22: -1}
                )
                # M3 = a(f-h)
                add_rank_unit(
                    {u_a: 1},
                    {v_f: 1, v_h: -1},
                    {w_c12: 1, w_c22: 1}
                )
                # M4 = d(g-e)
                add_rank_unit(
                    {u_d: 1},
                    {v_g: 1, v_e: -1},
                    {w_c11: 1, w_c21: 1}
                )
                # M5 = (a+b)h
                add_rank_unit(
                    {u_a: 1, u_b: 1},
                    {v_h: 1},
                    {w_c11: -1, w_c12: 1}
                )
                # M6 = (c-a)(e+f)
                add_rank_unit(
                    {u_c: 1, u_a: -1},
                    {v_e: 1, v_f: 1},
                    {w_c22: 1}
                )
                # M7 = (b-d)(g+h)
                add_rank_unit(
                    {u_b: 1, u_d: -1},
                    {v_g: 1, v_h: 1},
                    {w_c11: 1}
                )
                
    if col != rank:
        return None
        
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p = 96"""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, num_trials=3):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    rank = U.shape[1]
    
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        # Efficient verification using einsum
        # C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is tricky to do in one einsum because indices are mixed.
        # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        # Then C[i,k] = sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r]
        # einsum('ijr,jkr,kr->ik', U_reshaped, V_reshaped, W_reshaped)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        C_decomp = np.einsum('ijr,jkr,kr->ik', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(C_decomp, C_expected, atol=1e-9):
            return False
            
    return True
# EVOLVE-BLOCK-END
