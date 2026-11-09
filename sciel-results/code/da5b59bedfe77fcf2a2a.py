import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Block decomposition:
    # A (2x4) = [A0, A1] where A0, A1 are 2x2 (columns 0-1, 2-3)
    # B (4x6) split into 6 2x2 blocks: B[i,j] for i in {0,1} (row pair), j in {0,1,2} (col pair)
    # C (2x6) = [C0, C1, C2] where Ck = A0*B[0,k] + A1*B[1,k]
    # Each 2x2x2 product uses Strassen (rank 7), total rank = 6*7 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(i, j):
        return i * p + j
        
    def w_idx(i, j):
        return j * n + i

    col_offset = 0
    for a_block in range(2):  # A0 (cols 0-1) or A1 (cols 2-3)
        if time.time() > deadline:
            break
        a_col_base = a_block * 2
        
        for b_col_block in range(3):  # B col blocks 0-1, 2-3, 4-5
            if time.time() > deadline:
                break
            # B row block: a_block=0 -> rows 0-1, a_block=1 -> rows 2-3
            b_row_base = a_block * 2
            b_col_base = b_col_block * 2
            c_col_base = b_col_block * 2
            
            # Strassen 7 terms for 2x2x2
            # Local: A_sub[i,j], B_sub[i,j], C_sub[i,j]
            # A_sub[i,j] = A[i, a_col_base+j]
            # B_sub[i,j] = B[b_row_base+i, b_col_base+j]
            # C_sub[i,j] = C[i, c_col_base+j]
            
            # M1 = (a00+a11)(b00+b11): C00+, C11+
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # M2 = (a10+a11)b00: C10+, C11-
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] -= 1
            col_offset += 1
            
            # M3 = a00(b01-b11): C01+, C11+
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # M4 = a11(b10-b00): C00+, C10+
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
            # M5 = (a00+a01)b11: C00-, C01+
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # M6 = (a10-a00)(b00+b01): C11+
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 0), col_offset] -= 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # M7 = (a01-a11)(b10+b11): C00+
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] -= 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
