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

    return {"U": mat_u.tolist(), "V": mat_v.tolist(), "W": mat_w.tolist()}
# EVOLVE-BLOCK-END
