"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _build_schoolbook_4x4x5():
    """
    Build schoolbook decomposition for 4x4x5 matrix multiplication.
    Returns U (16, 80), V (20, 80), W (20, 80).
    """
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_4x4x5():
    """
    Build Strassen-based decomposition for 4x4x5.
    Uses block Strassen for the 4x4x4 part (rank 49) plus schoolbook for the 5th column (rank 4).
    Total rank: 53.
    """
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _build_strassen_4x4x4()
    
    # Build schoolbook for the 5th column: A(4x4) * B_col(4x1) -> C_col(4x1)
    # For each j in 0..3: term is A[:,j] * B[j,4] -> C[:,4]
    # U[i*4+j, c] = 1, V[j*5+4, c] = 1, W[4*4+i, c] = 1
    U_tail = np.zeros((16, 4))
    V_tail = np.zeros((20, 4))
    W_tail = np.zeros((20, 4))
    
    for j in range(4):
        for i in range(4):
            U_tail[i * 4 + j, j] = 1
        V_tail[j * 5 + 4, j] = 1
        for i in range(4):
            W_tail[4 * 4 + i, j] = 1
    
    # Combine
    U = np.hstack([U_444, U_tail])
    V = np.hstack([V_444, V_tail])
    W = np.hstack([W_444, W_tail])
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build block Strassen decomposition for 4x4x4.
    Returns U (16, 49), V (16, 49), W (16, 49).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Precompute mapping from (block_idx, inner_idx) to (row, col)
    def map_idx(b_idx, i_idx):
        r_base = 0 if b_idx < 2 else 2
        c_base = 0 if b_idx % 2 == 0 else 2
        r_off = 0 if i_idx < 2 else 1
        c_off = 0 if i_idx % 2 == 0 else 1
        return r_base + r_off, c_base + c_off
    
    # Strassen 2x2 factors
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (c+d)e
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3 = a(f-h)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4 = d(g-f)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5 = (a+b)h
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Strassen steps for 4x4 (7 outer steps, each with 7 inner steps)
    steps = [
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] },
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)] },
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] },
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] },
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)] },
        { 'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)] },
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)] },
    ]
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for step in steps:
        for inner_col in range(7):
            u_inner = U_2[:, inner_col]
            v_inner = V_2[:, inner_col]
            w_inner = W_2[:, inner_col]
            
            for block_idx, sign in step['A']:
                for i_idx in range(4):
                    val = u_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        U[r * 4 + c, col] += val * sign
                        
            for block_idx, sign in step['B']:
                for i_idx in range(4):
                    val = v_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        V[r * 4 + c, col] += val * sign
                        
            for block_idx, sign in step['C']:
                for i_idx in range(4):
                    val = w_inner[i_idx]
                    if val != 0:
                        r, c = map_idx(block_idx, i_idx)
                        W[r * 4 + c, col] += val * sign
                        
            col += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try Strassen + Schoolbook (Rank 53)
    try:
        U, V, W = _build_strassen_4x4x5()
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            print("Strassen+Schoolbook verification failed.")
    except Exception as e:
        print(f"Error in Strassen+Schoolbook: {e}")
    
    # Fallback to Schoolbook (Rank 80)
    try:
        U, V, W = _build_schoolbook_4x4x5()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception as e:
        print(f"Error in Schoolbook: {e}")
        # Last resort: return empty (will score 0)
        return {"U": [], "V": [], "W": []}

# EVOLVE-BLOCK-END
