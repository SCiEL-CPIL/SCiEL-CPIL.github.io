import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Local terms for 3x2x2 block
    # U: 6 entries (3 rows, 2 cols)
    # V: 4 entries (2 rows, 2 cols)
    # W: 6 entries (2 cols, 3 rows)
    
    local_terms = []
    
    # 1. M1 = (A00+A11)(B00+B11)
    local_terms.append(([1, 1, 0], [1, 0, 0, 1], [[1, 0, 0], [0, 1, 0]]))
    # 2. M2 = (A10+A11)B00
    local_terms.append(([0, 1, 1], [1, 0, 0, 0], [[0, 0, 0], [1, 0, 0]]))
    # 3. M3 = A00(B01-B11)
    local_terms.append(([1, 0, 0], [0, 1, 0, -1], [[0, 0, 0], [0, 1, 0]]))
    # 4. M4 = A11(B10-B00)
    local_terms.append(([0, 1, 0], [0, 0, 1, -1], [[0, 0, 0], [1, 1, 0]]))
    # 5. M5 = (A00+A01)B11
    local_terms.append(([1, 1, 0], [0, 0, 0, 1], [[0, 0, 0], [0, -1, 0]]))
    # 6. M6 = (A10-A00)(B00+B01)
    local_terms.append(([-1, 0, 1], [1, 1, 0, 0], [[0, 0, 0], [0, 1, 0]]))
    # 7. M7 = (A01-A11)(B10+B11)
    local_terms.append(([0, 1, -1], [0, 0, 1, 1], [[1, 0, 0], [0, 1, 0]]))
    
    # 8. M8 = A20 B00
    local_terms.append(([0, 0, 1], [1, 0, 0, 0], [[1, 0, 0], [0, 0, 0]]))
    # 9. M9 = A21 B11
    local_terms.append(([0, 0, 1], [0, 0, 0, 1], [[0, 0, 0], [1, 0, 0]]))
    # 10. M10 = A20 B01
    local_terms.append(([0, 0, 1], [0, 1, 0, 0], [[0, 0, 0], [1, 0, 0]]))
    # 11. M11 = A21 B10
    local_terms.append(([0, 0, 1], [0, 0, 1, 0], [[1, 0, 0], [0, 0, 0]]))
    
    rank = 8 * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 8 blocks
    # Block (a_block, b_block)
    # a_block 0: A cols 0,1. a_block 1: A cols 2,3.
    # b_block 0: B rows 0,1, cols 0,1.
    # b_block 1: B rows 0,1, cols 2,3.
    # b_block 2: B rows 0,1, cols 4,5.
    # b_block 3: B rows 0,1, cols 6,7.
    # b_block 4: B rows 2,3, cols 0,1.
    # b_block 5: B rows 2,3, cols 2,3.
    # b_block 6: B rows 2,3, cols 4,5.
    # b_block 7: B rows 2,3, cols 6,7.
    
    for a_block in range(2):
        for b_block in range(4):
            # Determine global indices
            a_col_offset = a_block * 2
            b_row_offset = (b_block // 2) * 2
            b_col_offset = (b_block % 2) * 2
            
            for u_loc, v_loc, w_loc in local_terms:
                # U
                for i in range(3):
                    for j in range(2):
                        if u_loc[i * 2 + j] != 0:
                            U[i * m + (a_col_offset + j)][col] = u_loc[i * 2 + j]
                
                # V
                for i in range(2):
                    for j in range(2):
                        if v_loc[i * 2 + j] != 0:
                            V[(b_row_offset + i) * p + (b_col_offset + j)][col] = v_loc[i * 2 + j]
                
                # W
                for k in range(2):
                    for i in range(3):
                        if w_loc[k][i] != 0:
                            W[(b_col_offset + k) * n + i][col] = w_loc[k][i]
                
                col += 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
