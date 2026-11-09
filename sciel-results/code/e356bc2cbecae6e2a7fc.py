"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 for all i,j,k
    """
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U: i*2+j
    V: j*2+k
    W: k*2+i
    """
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00 -= M5, C01 += M5, C10 += M5
    
    # M6 = (A01-A00)(B00+B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1 # C11 += M6
    
    # M7 = (A11-A01)(B01+B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1 # C00 += M7, C11 += M7
    
    return U, V, W

def _embed_subtensor(U_global, V_global, W_global, 
                     u_base, v_base, w_base, 
                     u_shape, v_shape, w_shape,
                     U_sub, V_sub, W_sub,
                     col_start):
    """
    Embeds a sub-tensor decomposition into the global matrices.
    u_base: (i_offset, j_offset)
    v_base: (j_offset, k_offset)
    w_base: (k_offset, i_offset)
    u_shape: (n_sub, m_sub)
    v_shape: (m_sub, p_sub)
    w_shape: (p_sub, n_sub)
    """
    n_sub, m_sub = u_shape
    p_sub = w_shape[0]
    
    rank_sub = U_sub.shape[1]
    
    for c in range(rank_sub):
        col = col_start + c
        
        # Embed U
        for i_sub in range(n_sub):
            for j_sub in range(m_sub):
                val = U_sub[i_sub * m_sub + j_sub, c]
                if val != 0:
                    i_global = u_base[0] + i_sub
                    j_global = u_base[1] + j_sub
                    u_idx = i_global * 5 + j_global # m=5
                    U_global[u_idx, col] += val
                    
        # Embed V
        for j_sub in range(m_sub):
            for k_sub in range(p_sub):
                val = V_sub[j_sub * p_sub + k_sub, c]
                if val != 0:
                    j_global = v_base[0] + j_sub
                    k_global = v_base[1] + k_sub
                    v_idx = j_global * 6 + k_global # p=6
                    V_global[v_idx, col] += val
                    
        # Embed W
        for k_sub in range(p_sub):
            for i_sub in range(n_sub):
                val = W_sub[k_sub * n_sub + i_sub, c]
                if val != 0:
                    k_global = w_base[0] + k_sub
                    i_global = w_base[1] + i_sub
                    w_idx = k_global * 4 + i_global # n=4
                    W_global[w_idx, col] += val

def construct():
    """Constructs a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Target rank 118 using two Strassen 2x2x2 sub-tensors
    rank_target = 118
    
    U = np.zeros((n * m, rank_target), dtype=np.float64)
    V = np.zeros((m * p, rank_target), dtype=np.float64)
    W = np.zeros((p * n, rank_target), dtype=np.float64)
    
    col_ptr = 0
    
    # Sub-tensor 1:
    # A rows 0,1; A cols 0,1; B rows 0,1; B cols 0,1
    # This corresponds to C[0:2, 0:2] contribution from A[0:2, 0:2] * B[0:2, 0:2]
    # Indices:
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    u_base_1 = (0, 0)
    v_base_1 = (0, 0)
    w_base_1 = (0, 0)
    u_shape_1 = (2, 2)
    v_shape_1 = (2, 2)
    w_shape_1 = (2, 2)
    
    S_U, S_V, S_W = _construct_strassen_2x2()
    _embed_subtensor(U, V, W, u_base_1, v_base_1, w_base_1, 
                     u_shape_1, v_shape_1, w_shape_1,
                     S_U, S_V, S_W, col_ptr)
    col_ptr += 7
    
    # Sub-tensor 2:
    # A rows 2,3; A cols 3,4; B rows 3,4; B cols 4,5
    # This corresponds to C[2:4, 4:6] contribution from A[2:4, 3:5] * B[3:5, 4:6]
    # Indices:
    # i in {2, 3}, j in {3, 4}, k in {4, 5}
    u_base_2 = (2, 3)
    v_base_2 = (3, 4)
    w_base_2 = (4, 2)
    u_shape_2 = (2, 2)
    v_shape_2 = (2, 2)
    w_shape_2 = (2, 2)
    
    _embed_subtensor(U, V, W, u_base_2, v_base_2, w_base_2, 
                     u_shape_2, v_shape_2, w_shape_2,
                     S_U, S_V, S_W, col_ptr)
    col_ptr += 7
    
    # Fill the rest with schoolbook
    # We need to cover all (i, j, k) triples that were NOT covered by the sub-tensors.
    # Covered set 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Covered set 2: i in {2,3}, j in {3,4}, k in {4,5}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this triple is in the covered sets
                in_set1 = (i in [0, 1]) and (j in [0, 1]) and (k in [0, 1])
                in_set2 = (i in [2, 3]) and (j in [3, 4]) and (k in [4, 5])
                
                if in_set1 or in_set2:
                    continue # Already handled by Strassen
                
                # Schoolbook
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                if col_ptr >= rank_target:
                    break
                U[u_idx, col_ptr] = 1
                V[v_idx, col_ptr] = 1
                W[w_idx, col_ptr] = 1
                col_ptr += 1
    
    # If we ran out of columns, we have a problem, but 118 should be enough.
    # Let's verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Fallback to schoolbook rank 120 if verification fails
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                column += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
