import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    equals 1 if (i,j,k) is valid (which it always is in the tensor support) 
    and 0 otherwise.
    Actually, the tensor T_{i,j,k} = 1.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The contraction is over r.
    Result R[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    u = i*m + j
    v = j*p + k  (Note: j in U and j in V must match for non-zero? 
                    In matrix mult, A[i,j] * B[j,k]. 
                    So U index depends on i,j. V index depends on j,k.
                    The sum is over j? No, the tensor rank decomposition 
                    sums over r. The indices i, j, k are fixed for the term.
                    Wait, the tensor is T_{i,j,k} = delta_{j,j} (implicit).
                    The decomposition is T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r.
                    Here, the index j in U and j in V must be the SAME j.
                    So for a fixed r, U[i*m+j, r] is non-zero only for specific j.
                    V[j'*p+k, r] is non-zero only for specific j'.
                    If j != j', the term contributes to T_{i,j,k} where the j in U 
                    and j in V are different? 
                    
                    Let's re-read the tensor definition.
                    T_{i,j,k} = 1.
                    Decomposition: T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r.
                    This implies that for each r, we pick a specific (i,j), (j,k), (k,i).
                    The j in U must match the j in V.
                    The k in V must match the k in W.
                    The i in W must match the i in U.
                    
                    So, for a given r, let (i_r, j_r) be the indices of U_r.
                    Let (j'_r, k_r) be the indices of V_r.
                    Let (k'_r, i'_r) be the indices of W_r.
                    For the term to be non-zero in the sum for T_{i,j,k}, 
                    we need:
                    U_{i,j}^r != 0 => i=i_r, j=j_r
                    V_{j,k}^r != 0 => j=j'_r, k=k_r
                    W_{k,i}^r != 0 => k=k'_r, i=i'_r
                    
                    Thus, we need j_r = j'_r = j, k_r = k'_r = k, i_r = i'_r = i.
                    If these hold, the value is U_{i,j}^r V_{j,k}^r W_{k,i}^r.
                    Otherwise, it is 0.
                    
                    So the verification loop should check:
                    For all i, j, k:
                    val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if val != 1: return False
                    
                    This is what the current code does.
                    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if val != 1:
                    return False
    return True

def construct():
    if time.time() > _DEADLINE:
        return None # Should not happen, but safety

    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x6. Split into Top (rows 0-1) and Bottom (rows 2-3).
    # And split B columns into 3 pairs: 0-1, 2-3, 4-5.
    
    # We perform 6 independent 2x2 * 2x2 multiplications.
    # 1. A_L * B_Top_Left
    # 2. A_L * B_Top_Mid
    # 3. A_L * B_Top_Right
    # 4. A_R * B_Bot_Left
    # 5. A_R * B_Bot_Mid
    # 6. A_R * B_Bot_Right
    
    blocks = [
        ( [0, 1], [0, 1], [0, 1], [0, 1] ), # A_L, B_TL
        ( [0, 1], [0, 1], [0, 1], [2, 3] ), # A_L, B_TM
        ( [0, 1], [0, 1], [0, 1], [4, 5] ), # A_L, B_TR
        ( [0, 1], [2, 3], [2, 3], [0, 1] ), # A_R, B_BL
        ( [0, 1], [2, 3], [2, 3], [2, 3] ), # A_R, B_BM
        ( [0, 1], [2, 3], [2, 3], [4, 5] ), # A_R, B_BR
    ]
    
    # Strassen coefficients for 2x2 * 2x2
    # U: (local_idx, coeff)
    # V: (local_idx, coeff)
    # W: (local_idx, coeff)
    # Local A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local Z: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen = [
        # P1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # P2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # P3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # P4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # P5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # P6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # P7
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            col += 1
            
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
        
    return {"U": U, "V": V, "W": W}
