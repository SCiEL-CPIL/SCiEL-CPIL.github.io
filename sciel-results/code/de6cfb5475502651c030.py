import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 140
    
    rows, mid, cols = DIMENSIONS
    
    # Total rank 50
    # Structure:
    # 4 * Strassen(2,2,2) for 2x4x4 -> 28
    # 2 * (Strassen(2,2,2) + Schoolbook(2,2,1)) for 2x4x3 -> 2*(7+4)=22
    # Total 50
    
    r = 50
    mat_u = np.zeros((rows * mid, r), dtype=np.float64)
    mat_v = np.zeros((mid * cols, r), dtype=np.float64)
    mat_w = np.zeros((cols * rows, r), dtype=np.float64)
    
    idx = 0
    
    def add_strassen_222(i_start, j_start, k_start):
        nonlocal idx
        # Indices:
        # A: (i_start, j_start) to (i_start+1, j_start+1)
        # B: (j_start, k_start) to (j_start+1, k_start+1)
        # C: (i_start, k_start) to (i_start+1, k_start+1)
        
        # M1
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M2
        mat_u[(i_start+1)*mid + j_start, idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = -1.0
        idx += 1
        
        # M3
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_v[j_start*cols + (k_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = -1.0
        mat_w[(k_start+1)*rows + i_start, idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M4
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = -1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1
        
        # M5
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_u[i_start*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = -1.0
        mat_w[(k_start+1)*rows + i_start, idx] = 1.0
        idx += 1
        
        # M6
        mat_u[(i_start+1)*mid + j_start, idx] = 1.0
        mat_u[i_start*mid + j_start, idx] = -1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_v[j_start*cols + (k_start+1), idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M7
        mat_u[i_start*mid + (j_start+1), idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = -1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1

    def add_schoolbook_221(i_start, j_start, k_start):
        nonlocal idx
        # A: 2x2, B: 2x1, C: 2x1
        # T1: A00 * B0
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1
        # T2: A01 * B1
        mat_u[i_start*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1
        # T3: A10 * B0
        mat_u[(i_start+1)*mid + j_start, idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        idx += 1
        # T4: A11 * B1
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        idx += 1

    # 2x4x4 part: 4 blocks of 2x2x2
    for j_blk in range(2):
        for k_blk in range(2):
            add_strassen_222(0, j_blk*2, k_blk*2)
            
    # 2x4x3 part: 2 blocks of (2x2x2 + 2x2x1)
    for j_blk in range(2):
        # First 2 cols (k=4,5)
        add_strassen_222(0, j_blk*2, 4)
        # Last 1 col (k=6)
        add_schoolbook_221(0, j_blk*2, 6)

    # Verify the decomposition
    U = mat_u
    V = mat_v
    W = mat_w
    
    # Create a random test matrix
    A = np.random.randn(rows, mid)
    B = np.random.randn(mid, cols)
    C_expected = A @ B
    
    # Compute using the decomposition
    C_decomposed = np.zeros((rows, cols))
    for r_idx in range(r):
        u_col = U[:, r_idx]
        v_col = V[:, r_idx]
        w_col = W[:, r_idx]
        
        # Reshape u_col to (rows, mid)
        u_mat = u_col.reshape(rows, mid)
        # Reshape v_col to (mid, cols)
        v_mat = v_col.reshape(mid, cols)
        # Reshape w_col to (cols, rows)
        w_mat = w_col.reshape(cols, rows)
        
        # The contribution is trace(A * u_mat * B * v_mat * w_mat)
        # Or more precisely: sum_{i,j,k} A[i,j] u_mat[i,j] v_mat[j,k] w_mat[k,i]
        # Let's compute it directly
        contrib = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(mid):
                for k in range(cols):
                    contrib[i, k] += A[i, j] * u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        C_decomposed += contrib
    
    # Check if the decomposition is correct
    if not np.allclose(C_expected, C_decomposed, atol=1e-10):
        # Fall back to schoolbook if the decomposition is incorrect
        r = rows * mid * cols
        mat_u = np.zeros((rows * mid, r), dtype=np.float64)
        mat_v = np.zeros((mid * cols, r), dtype=np.float64)
        mat_w = np.zeros((cols * rows, r), dtype=np.float64)
        
        idx = 0
        for i in range(rows):
            for j in range(mid):
                for k in range(cols):
                    mat_u[i*mid + j, idx] = 1.0
                    mat_v[j*cols + k, idx] = 1.0
                    mat_w[k*rows + i, idx] = 1.0
                    idx += 1
        
        return {"U": mat_u.tolist(), "V": mat_v.tolist(), "W": mat_w.tolist()}
    
    return {"U": mat_u.tolist(), "V": mat_v.tolist(), "W": mat_w.tolist()}
# EVOLVE-BLOCK-END
