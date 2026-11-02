"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 114 decomposition:
    # 2x (4x4x4 Strassen rank 49) + 1x (4x4x1 Schoolbook rank 16)
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_2x2_strassen(x_entries, y_entries, w_map, offset):
        """
        Adds 7 rank-1 terms for 2x2 matrix multiplication.
        x_entries: list of 4 lists of (u_idx, coeff) for X[0..3]
        y_entries: list of 4 lists of (v_idx, coeff) for Y[0..3]
        w_map: list of 4 lists of (w_idx, coeff) for Z[0..3]
        """
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        u_vec = {}
        for idx, c in x_entries[0]: u_vec[idx] = u_vec.get(idx, 0) + c
        for idx, c in x_entries[3]: u_vec[idx] = u_vec.get(idx, 0) + c
        
        v_vec = {}
        for idx, c in y_entries[0]: v_vec[idx] = v_vec.get(idx, 0) + c
        for idx, c in y_entries[3]: v_vec[idx] = v_vec.get(idx, 0) + c
        
        w_vec = {}
        for idx, c in w_map[0]: w_vec[idx] = w_vec.get(idx, 0) + c
        for idx, c in w_map[3]: w_vec[idx] = w_vec.get(idx, 0) + c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M2 = (X10 + X11) Y00 -> Z10 += 1
        u_vec = {}
        for idx, c in x_entries[2]: u_vec[idx] = u_vec.get(idx, 0) + c
        for idx, c in x_entries[3]: u_vec[idx] = u_vec.get(idx, 0) + c
        
        v_vec = {}
        for idx, c in y_entries[0]: v_vec[idx] = v_vec.get(idx, 0) + c
        
        w_vec = {}
        for idx, c in w_map[2]: w_vec[idx] = w_vec.get(idx, 0) + c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M3 = X00 (Y01 - Y11) -> Z01 += 1
        u_vec = {}
        for idx, c in x_entries[0]: u_vec[idx] = u_vec.get(idx, 0) + c
        
        v_vec = {}
        for idx, c in y_entries[1]: v_vec[idx] = v_vec.get(idx, 0) + c
        for idx, c in y_entries[3]: v_vec[idx] = v_vec.get(idx, 0) - c
        
        w_vec = {}
        for idx, c in w_map[1]: w_vec[idx] = w_vec.get(idx, 0) + c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M4 = X11 (Y10 - Y00) -> Z00 += 1, Z10 += 1
        u_vec = {}
        for idx, c in x_entries[3]: u_vec[idx] = u_vec.get(idx, 0) + c
        
        v_vec = {}
        for idx, c in y_entries[2]: v_vec[idx] = v_vec.get(idx, 0) + c
        for idx, c in y_entries[0]: v_vec[idx] = v_vec.get(idx, 0) - c
        
        w_vec = {}
        for idx, c in w_map[0]: w_vec[idx] = w_vec.get(idx, 0) + c
        for idx, c in w_map[2]: w_vec[idx] = w_vec.get(idx, 0) + c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M5 = (X00 + X01) Y11 -> Z01 += 1, Z00 -= 1
        u_vec = {}
        for idx, c in x_entries[0]: u_vec[idx] = u_vec.get(idx, 0) + c
        for idx, c in x_entries[1]: u_vec[idx] = u_vec.get(idx, 0) + c
        
        v_vec = {}
        for idx, c in y_entries[3]: v_vec[idx] = v_vec.get(idx, 0) + c
        
        w_vec = {}
        for idx, c in w_map[1]: w_vec[idx] = w_vec.get(idx, 0) + c
        for idx, c in w_map[0]: w_vec[idx] = w_vec.get(idx, 0) - c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M6 = (X10 - X00) (Y00 + Y01) -> Z11 += 1
        u_vec = {}
        for idx, c in x_entries[2]: u_vec[idx] = u_vec.get(idx, 0) + c
        for idx, c in x_entries[0]: u_vec[idx] = u_vec.get(idx, 0) - c
        
        v_vec = {}
        for idx, c in y_entries[0]: v_vec[idx] = v_vec.get(idx, 0) + c
        for idx, c in y_entries[1]: v_vec[idx] = v_vec.get(idx, 0) + c
        
        w_vec = {}
        for idx, c in w_map[3]: w_vec[idx] = w_vec.get(idx, 0) + c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        # M7 = (X01 - X11) (Y10 + Y11) -> Z00 += 1, Z11 -= 1
        u_vec = {}
        for idx, c in x_entries[1]: u_vec[idx] = u_vec.get(idx, 0) + c
        for idx, c in x_entries[3]: u_vec[idx] = u_vec.get(idx, 0) - c
        
        v_vec = {}
        for idx, c in y_entries[2]: v_vec[idx] = v_vec.get(idx, 0) + c
        for idx, c in y_entries[3]: v_vec[idx] = v_vec.get(idx, 0) + c
        
        w_vec = {}
        for idx, c in w_map[0]: w_vec[idx] = w_vec.get(idx, 0) + c
        for idx, c in w_map[3]: w_vec[idx] = w_vec.get(idx, 0) - c
        
        for idx, c in u_vec.items(): U[idx][offset] += c
        for idx, c in v_vec.items(): V[idx][offset] += c
        for idx, c in w_vec.items(): W[idx][offset] += c
        offset += 1
        
        return offset

    # Helper to get index in U, V, W
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 9 + k
    def w_idx(k, i): return k * 4 + i

    # Block 1: B cols 0-3
    # A blocks
    A00 = [(u_idx(0,0), 1), (u_idx(0,1), 1), (u_idx(1,0), 1), (u_idx(1,1), 1)] # This is wrong, A00 is 2x2
    # A00 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    # We need x_entries for X. X is a 2x2 matrix.
    # X[0,0] = A[0,0] ...
    
    # Let's define the linear forms for the 7 block products.
    
    # P1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    x1 = [
        [(u_idx(0,0), 1), (u_idx(2,2), 1)],
        [(u_idx(0,1), 1), (u_idx(2,3), 1)],
        [(u_idx(1,0), 1), (u_idx(3,2), 1)],
        [(u_idx(1,1), 1), (u_idx(3,3), 1)]
    ]
    
    # Y = B00 + B11
    # Y[0,0] = B[0,0] + B[2,0] (Note: B is 4x4 block, indices j,k)
    # B00 is B[0:2, 0:2]. B11 is B[2:4, 2:4].
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    y1 = [
        [(v_idx(0,0), 1), (v_idx(2,2), 1)],
        [(v_idx(0,1), 1), (v_idx(2,3), 1)],
        [(v_idx(1,0), 1), (v_idx(3,2), 1)],
        [(v_idx(1,1), 1), (v_idx(3,3), 1)]
    ]
    
    # W map for P1:
    # P1 contributes to C00 (top-left) and C11 (bottom-right)
    # C00 indices: C[0,0], C[0,1], C[1,0], C[1,1] -> w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    # C11 indices: C[2,2], C[2,3], C[3,2], C[3,3] -> w_idx(2,2), w_idx(3,2), w_idx(2,3), w_idx(3,3)
    # Z[0,0] -> C[0,0] and C[2,2]
    # Z[0,1] -> C[0,1] and C[2,3]
    # Z[1,0] -> C[1,0] and C[3,2]
    # Z[1,1] -> C[1,1] and C[3,3]
    
    w1 = [
        [(w_idx(0,0), 1), (w_idx(2,2), 1)],
        [(w_idx(1,0), 1), (w_idx(2,3), 1)],
        [(w_idx(0,1), 1), (w_idx(3,2), 1)],
        [(w_idx(1,1), 1), (w_idx(3,3), 1)]
    ]
    
    offset = 0
    offset = add_2x2_strassen(x1, y1, w1, offset)
    
    # P2 = (A10 + A11) B00
    # X = A10 + A11
    # X[0,0] = A[2,0] + A[2,2]
    # X[0,1] = A[2,1] + A[2,3]
    # X[1,0] = A[3,0] + A[3,2]
    # X[1,1] = A[3,1] + A[3,3]
    
    x2 = [
        [(u_idx(2,0), 1), (u_idx(2,2), 1)],
        [(u_idx(2,1), 1), (u_idx(2,3), 1)],
        [(u_idx(3,0), 1), (u_idx(3,2), 1)],
        [(u_idx(3,1), 1), (u_idx(3,3), 1)]
    ]
    
    # Y = B00
    y2 = [
        [(v_idx(0,0), 1)],
        [(v_idx(0,1), 1)],
        [(v_idx(1,0), 1)],
        [(v_idx(1,1), 1)]
    ]
    
    # W map for P2:
    # P2 contributes to C10 (bottom-left)
    # C10 indices: C[2,0], C[2,1], C[3,0], C[3,1]
    # Z[0,0] -> C[2,0]
    # Z[0,1] -> C[2,1]
    # Z[1,0] -> C[3,0]
    # Z[1,1] -> C[3,1]
    
    w2 = [
        [(w_idx(0,2), 1)],
        [(w_idx(1,2), 1)],
        [(w_idx(0,3), 1)],
        [(w_idx(1,3), 1)]
    ]
    
    offset = add_2x2_strassen(x2, y2, w2, offset)
    
    # P3 = A00 (B01 - B11)
    # X = A00
    x3 = [
        [(u_idx(0,0), 1)],
        [(u_idx(0,1), 1)],
        [(u_idx(1,0), 1)],
        [(u_idx(1,1), 1)]
    ]
    
    # Y = B01 - B11
    # Y[0,0] = B[0,2] - B[2,2]
    # Y[0,1] = B[0,3] - B[2,3]
    # Y[1,0] = B[1,2] - B[3,2]
    # Y[1,1] = B[1,3] - B[3,3]
    
    y3 = [
        [(v_idx(0,2), 1), (v_idx(2,2), -1)],
        [(v_idx(0,3), 1), (v_idx(2,3), -1)],
        [(v_idx(1,2), 1), (v_idx(3,2), -1)],
        [(v_idx(1,3), 1), (v_idx(3,3), -1)]
    ]
    
    # W map for P3:
    # P3 contributes to C01 (top-right)
    # C01 indices: C[0,2], C[0,3], C[1,2], C[1,3]
    # Z[0,0] -> C[0,2]
    # Z[0,1] -> C[0,3]
    # Z[1,0] -> C[1,2]
    # Z[1,1] -> C[1,3]
    
    w3 = [
        [(w_idx(2,0), 1)],
        [(w_idx(3,0), 1)],
        [(w_idx(2,1), 1)],
        [(w_idx(3,1), 1)]
    ]
    
    offset = add_2x2_strassen(x3, y3, w3, offset)
    
    # P4 = A11 (B10 - B00)
    # X = A11
    x4 = [
        [(u_idx(2,2), 1)],
        [(u_idx(2,3), 1)],
        [(u_idx(3,2), 1)],
        [(u_idx(3,3), 1)]
    ]
    
    # Y = B10 - B00
    # Y[0,0] = B[2,0] - B[0,0]
    # Y[0,1] = B[2,1] - B[0,1]
    # Y[1,0] = B[3,0] - B[1,0]
    # Y[1,1] = B[3,1] - B[1,1]
    
    y4 = [
        [(v_idx(2,0), 1), (v_idx(0,0), -1)],
        [(v_idx(2,1), 1), (v_idx(0,1), -1)],
        [(v_idx(3,0), 1), (v_idx(1,0), -1)],
        [(v_idx(3,1), 1), (v_idx(1,1), -1)]
    ]
    
    # W map for P4:
    # P4 contributes to C00 (top-left) and C10 (bottom-left)
    # C00: C[0,0], C[0,1], C[1,0], C[1,1]
    # C10: C[2,0], C[2,1], C[3,0], C[3,1]
    # Z[0,0] -> C[0,0], C[2,0]
    # Z[0,1] -> C[0,1], C[2,1]
    # Z[1,0] -> C[1,0], C[3,0]
    # Z[1,1] -> C[1,1], C[3,1]
    
    w4 = [
        [(w_idx(0,0), 1), (w_idx(0,2), 1)],
        [(w_idx(1,0), 1), (w_idx(1,2), 1)],
        [(w_idx(0,1), 1), (w_idx(0,3), 1)],
        [(w_idx(1,1), 1), (w_idx(1,3), 1)]
    ]
    
    offset = add_2x2_strassen(x4, y4, w4, offset)
    
    # P5 = (A00 + A01) B11
    # X = A00 + A01
    # X[0,0] = A[0,0] + A[0,2]
    # X[0,1] = A[0,1] + A[0,3]
    # X[1,0] = A[1,0] + A[1,2]
    # X[1,1] = A[1,1] + A[1,3]
    
    x5 = [
        [(u_idx(0,0), 1), (u_idx(0,2), 1)],
        [(u_idx(0,1), 1), (u_idx(0,3), 1)],
        [(u_idx(1,0), 1), (u_idx(1,2), 1)],
        [(u_idx(1,1), 1), (u_idx(1,3), 1)]
    ]
    
    # Y = B11
    y5 = [
        [(v_idx(2,2), 1)],
        [(v_idx(2,3), 1)],
        [(v_idx(3,2), 1)],
        [(v_idx(3,3), 1)]
    ]
    
    # W map for P5:
    # P5 contributes to C01 (top-right) and C00 (top-left, with -1)
    # C01: C[0,2], C[0,3], C[1,2], C[1,3]
    # C00: C[0,0], C[0,1], C[1,0], C[1,1]
    # Z[0,0] -> C[0,2], C[0,0](-1)
    # Z[0,1] -> C[0,3], C[0,1](-1)
    # Z[1,0] -> C[1,2], C[1,0](-1)
    # Z[1,1] -> C[1,3], C[1,1](-1)
    
    w5 = [
        [(w_idx(2,0), 1), (w_idx(0,0), -1)],
        [(w_idx(3,0), 1), (w_idx(1,0), -1)],
        [(w_idx(2,1), 1), (w_idx(0,1), -1)],
        [(w_idx(3,1), 1), (w_idx(1,1), -1)]
    ]
    
    offset = add_2x2_strassen(x5, y5, w5, offset)
    
    # P6 = (A10 - A00) (B00 + B01)
    # X = A10 - A00
    # X[0,0] = A[2,0] - A[0,0]
    # X[0,1] = A[2,1] - A[0,1]
    # X[1,0] = A[3,0] - A[1,0]
    # X[1,1] = A[3,1] - A[1,1]
    
    x6 = [
        [(u_idx(2,0), 1), (u_idx(0,0), -1)],
        [(u_idx(2,1), 1), (u_idx(0,1), -1)],
        [(u_idx(3,0), 1), (u_idx(1,0), -1)],
        [(u_idx(3,1), 1), (u_idx(1,1), -1)]
    ]
    
    # Y = B00 + B01
    # Y[0,0] = B[0,0] + B[0,2]
    # Y[0,1] = B[0,1] + B[0,3]
    # Y[1,0] = B[1,0] + B[1,2]
    # Y[1,1] = B[1,1] + B[1,3]
    
    y6 = [
        [(v_idx(0,0), 1), (v_idx(0,2), 1)],
        [(v_idx(0,1), 1), (v_idx(0,3), 1)],
        [(v_idx(1,0), 1), (v_idx(1,2), 1)],
        [(v_idx(1,1), 1), (v_idx(1,3), 1)]
    ]
    
    # W map for P6:
    # P6 contributes to C11 (bottom-right)
    # C11: C[2,2], C[2,3], C[3,2], C[3,3]
    # Z[0,0] -> C[2,2]
    # Z[0,1] -> C[2,3]
    # Z[1,0] -> C[3,2]
    # Z[1,1] -> C[3,3]
    
    w6 = [
        [(w_idx(2,2), 1)],
        [(w_idx(3,2), 1)],
        [(w_idx(2,3), 1)],
        [(w_idx(3,3), 1)]
    ]
    
    offset = add_2x2_strassen(x6, y6, w6, offset)
    
    # P7 = (A01 - A11) (B10 + B11)
    # X = A01 - A11
    # X[0,0] = A[0,2] - A[2,2]
    # X[0,1] = A[0,3] - A[2,3]
    # X[1,0] = A[1,2] - A[3,2]
    # X[1,1] = A[1,3] - A[3,3]
    
    x7 = [
        [(u_idx(0,2), 1), (u_idx(2,2), -1)],
        [(u_idx(0,3), 1), (u_idx(2,3), -1)],
        [(u_idx(1,2), 1), (u_idx(3,2), -1)],
        [(u_idx(1,3), 1), (u_idx(3,3), -1)]
    ]
    
    # Y = B10 + B11
    # Y[0,0] = B[2,0] + B[2,2]
    # Y[0,1] = B[2,1] + B[2,3]
    # Y[1,0] = B[3,0] + B[3,2]
    # Y[1,1] = B[3,1] + B[3,3]
    
    y7 = [
        [(v_idx(2,0), 1), (v_idx(2,2), 1)],
        [(v_idx(2,1), 1), (v_idx(2,3), 1)],
        [(v_idx(3,0), 1), (v_idx(3,2), 1)],
        [(v_idx(3,1), 1), (v_idx(3,3), 1)]
    ]
    
    # W map for P7:
    # P7 contributes to C00 (top-left) and C11 (bottom-right, with -1)
    # C00: C[0,0], C[0,1], C[1,0], C[1,1]
    # C11: C[2,2], C[2,3], C[3,2], C[3,3]
    # Z[0,0] -> C[0,0], C[2,2](-1)
    # Z[0,1] -> C[0,1], C[2,3](-1)
    # Z[1,0] -> C[1,0], C[3,2](-1)
    # Z[1,1] -> C[1,1], C[3,3](-1)
    
    w7 = [
        [(w_idx(0,0), 1), (w_idx(2,2), -1)],
        [(w_idx(1,0), 1), (w_idx(3,2), -1)],
        [(w_idx(0,1), 1), (w_idx(2,3), -1)],
        [(w_idx(1,1), 1), (w_idx(3,3), -1)]
    ]
    
    offset = add_2x2_strassen(x7, y7, w7, offset)
    
    # Block 2: B cols 4-7
    # Similar to Block 1, but V indices shifted by 4 in k, W indices shifted by 4 in k.
    
    # P1' = (A00 + A11)(B00' + B11')
    # B00' is B[0:2, 4:6]. B11' is B[2:4, 6:8].
    
    x1p = x1 # Same X
    y1p = [
        [(v_idx(0,4), 1), (v_idx(2,6), 1)],
        [(v_idx(0,5), 1), (v_idx(2,7), 1)],
        [(v_idx(1,4), 1), (v_idx(3,6), 1)],
        [(v_idx(1,5), 1), (v_idx(3,7), 1)]
    ]
    
    w1p = [
        [(w_idx(4,0), 1), (w_idx(6,2), 1)],
        [(w_idx(5,0), 1), (w_idx(6,3), 1)],
        [(w_idx(4,1), 1), (w_idx(7,2), 1)],
        [(w_idx(5,1), 1), (w_idx(7,3), 1)]
    ]
    
    offset = add_2x2_strassen(x1p, y1p, w1p, offset)
    
    # P2' = (A10 + A11) B00'
    x2p = x2
    y2p = [
        [(v_idx(0,4), 1)],
        [(v_idx(0,5), 1)],
        [(v_idx(1,4), 1)],
        [(v_idx(1,5), 1)]
    ]
    w2p = [
        [(w_idx(4,2), 1)],
        [(w_idx(5,2), 1)],
        [(w_idx(4,3), 1)],
        [(w_idx(5,3), 1)]
    ]
    offset = add_2x2_strassen(x2p, y2p, w2p, offset)
    
    # P3' = A00 (B01' - B11')
    x3p = x3
    y3p = [
        [(v_idx(0,6), 1), (v_idx(2,6), -1)],
        [(v_idx(0,7), 1), (v_idx(2,7), -1)],
        [(v_idx(1,6), 1), (v_idx(3,6), -1)],
        [(v_idx(1,7), 1), (v_idx(3,7), -1)]
    ]
    w3p = [
        [(w_idx(6,0), 1)],
        [(w_idx(7,0), 1)],
        [(w_idx(6,1), 1)],
        [(w_idx(7,1), 1)]
    ]
    offset = add_2x2_strassen(x3p, y3p, w3p, offset)
    
    # P4' = A11 (B10' - B00')
    x4p = x4
    y4p = [
        [(v_idx(2,4), 1), (v_idx(0,4), -1)],
        [(v_idx(2,5), 1), (v_idx(0,5), -1)],
        [(v_idx(3,4), 1), (v_idx(1,4), -1)],
        [(v_idx(3,5), 1), (v_idx(1,5), -1)]
    ]
    w4p = [
        [(w_idx(4,0), 1), (w_idx(4,2), 1)],
        [(w_idx(5,0), 1), (w_idx(5,2), 1)],
        [(w_idx(4,1), 1), (w_idx(4,3), 1)],
        [(w_idx(5,1), 1), (w_idx(5,3), 1)]
    ]
    offset = add_2x2_strassen(x4p, y4p, w4p, offset)
    
    # P5' = (A00 + A01) B11'
    x5p = x5
    y5p = [
        [(v_idx(2,6), 1)],
        [(v_idx(2,7), 1)],
        [(v_idx(3,6), 1)],
        [(v_idx(3,7), 1)]
    ]
    w5p = [
        [(w_idx(6,0), 1), (w_idx(4,0), -1)],
        [(w_idx(7,0), 1), (w_idx(5,0), -1)],
        [(w_idx(6,1), 1), (w_idx(4,1), -1)],
        [(w_idx(7,1), 1), (w_idx(5,1), -1)]
    ]
    offset = add_2x2_strassen(x5p, y5p, w5p, offset)
    
    # P6' = (A10 - A00) (B00' + B01')
    x6p = x6
    y6p = [
        [(v_idx(0,4), 1), (v_idx(0,6), 1)],
        [(v_idx(0,5), 1), (v_idx(0,7), 1)],
        [(v_idx(1,4), 1), (v_idx(1,6), 1)],
        [(v_idx(1,5), 1), (v_idx(1,7), 1)]
    ]
    w6p = [
        [(w_idx(6,2), 1)],
        [(w_idx(7,2), 1)],
        [(w_idx(6,3), 1)],
        [(w_idx(7,3), 1)]
    ]
    offset = add_2x2_strassen(x6p, y6p, w6p, offset)
    
    # P7' = (A01 - A11) (B10' + B11')
    x7p = x7
    y7p = [
        [(v_idx(2,4), 1), (v_idx(2,6), 1)],
        [(v_idx(2,5), 1), (v_idx(2,7), 1)],
        [(v_idx(3,4), 1), (v_idx(3,6), 1)],
        [(v_idx(3,5), 1), (v_idx(3,7), 1)]
    ]
    w7p = [
        [(w_idx(4,0), 1), (w_idx(6,2), -1)],
        [(w_idx(5,0), 1), (w_idx(7,2), -1)],
        [(w_idx(4,1), 1), (w_idx(6,3), -1)],
        [(w_idx(5,1), 1), (w_idx(7,3), -1)]
    ]
    offset = add_2x2_strassen(x7p, y7p, w7p, offset)
    
    # Block 3: B col 8 (Schoolbook)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][offset] = 1
            V[j * 9 + 8][offset] = 1
            W[8 * 4 + i][offset] = 1
            offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
