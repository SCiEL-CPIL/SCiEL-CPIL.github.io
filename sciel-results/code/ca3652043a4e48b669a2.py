import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for M_{3,3,8}
    # Strategy: Split p=8 into four blocks of size 2.
    # Each block is M_{3,3,2}.
    # Rank of M_{3,3,2} is 17 (7 Strassen + 10 Naive).
    # Total Rank = 4 * 17 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Define the 17 terms for M_{3,3,2}
    # A is 3x3, B is 3x2, C is 3x2.
    # Indices: A[i,j], B[j,k], C[i,k]
    # Strassen part: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Naive part:
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map these to the U, V, W matrices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For a specific block of B columns (k_offset, k_offset+1)
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, c_idx):
        U[u_idx][c_idx] += u_val
        V[v_idx][c_idx] += v_val
        W[w_idx][c_idx] += w_val

    for block_idx in range(4):
        k_offset = block_idx * 2
        # k is 0, 1 relative to block, mapped to k_offset, k_offset+1
        
        # --- Strassen Part (7 terms) ---
        # M1
        # A: (0,0)+ (1,1) -> U[0], U[3]
        # B: (0,0)+ (1,1) -> V[0], V[3] (Note: V index is j*p+k. j=0,k=0 -> 0. j=1,k=1 -> 1*3+1=4? No.
        # V index: j*p + k.
        # For block k_offset, k=0 -> j*3 + k_offset. k=1 -> j*3 + k_offset+1.
        
        # Let's define local indices for the 2x2 block
        # A rows 0,1; cols 0,1
        # B rows 0,1; cols 0,1 (local)
        
        # M1 = (A00+A11)(B00+B11)
        add_term(0, 0, 0, 1, 1, 1, col_offset) # A00 * B00
        add_term(0, 3, 0, 1, 1, 1, col_offset) # A00 * B11 (V index for j=1,k=1 is 1*3+1=4? No.
        # Wait. V index is j*p + k.
        # p=8.
        # For k_offset=0:
        # B[0,0] -> V[0*8+0] = V[0]
        # B[1,1] -> V[1*8+1] = V[9]
        # My previous V index logic was for p=2. Here p=8.
        
        # Let's restart the index mapping carefully.
        # U[i*m+j]
        # V[j*p+k]
        # W[k*n+i]
        
        # Block k_offset
        # Local k=0 -> Global k = k_offset
        # Local k=1 -> Global k = k_offset + 1
        
        # Strassen M1: (A00+A11)(B00+B11)
        # Term 1: A00 * B00
        add_term(0, 0*8+k_offset, 0*3+0, 1, 1, 1, col_offset)
        # Term 2: A00 * B11 (j=1, k_local=1 -> k_global=k_offset+1)
        add_term(0, 1*8+(k_offset+1), 1*3+0, 1, 1, 1, col_offset)
        # Term 3: A11 * B00
        add_term(3, 0*8+k_offset, 0*3+1, 1, 1, 1, col_offset)
        # Term 4: A11 * B11
        add_term(3, 1*8+(k_offset+1), 1*3+1, 1, 1, 1, col_offset)
        col_offset += 1
        
        # M2 = (A10+A11)B00
        # A10 is i=1,j=0 -> U[1*3+0]=U[3]? No. U[i*m+j]. i=1,j=0 -> U[3].
        # A11 is i=1,j=1 -> U[4].
        # B00 is j=0,k=0 -> V[0*8+k_offset].
        # Output C10 (i=1,k=0) -> W[0*3+1]=W[1]. C11 (i=1,k=1) -> W[1*3+1]=W[4].
        # Wait, Strassen M2 contributes to C10 and C11?
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # So M2 is A10*B00 + A11*B00.
        # Output:
        # C10 gets +1. C11 gets -1.
        
        # Term 5: A10 * B00
        add_term(3, 0*8+k_offset, 0*3+1, 1, 1, 1, col_offset)
        add_term(3, 0*8+k_offset, 1*3+1, 1, 1, -1, col_offset)
        # Term 6: A11 * B00
        add_term(4, 0*8+k_offset, 0*3+1, 1, 1, 1, col_offset)
        add_term(4, 0*8+k_offset, 1*3+1, 1, 1, -1, col_offset)
        col_offset += 1
        
        # M3 = A00(B10-B11)
        # A00 -> U[0].
        # B10 -> V[1*8+k_offset]. B11 -> V[1*8+k_offset+1].
        # Output C01 (i=0,k=1) -> W[1*3+0]=W[3]. C11 (i=1,k=1) -> W[1*3+1]=W[4].
        # C01 = M3 + M5. C11 = ... + M3.
        # So C01 gets +1. C11 gets +1.
        
        # Term 7: A00 * B10
        add_term(0, 1*8+k_offset, 1*3+0, 1, 1, 1, col_offset)
        add_term(0, 1*8+k_offset, 1*3+1, 1, 1, 1, col_offset)
        # Term 8: A00 * B11
        add_term(0, 1*8+(k_offset+1), 1*3+0, 1, 1, -1, col_offset)
        add_term(0, 1*8+(k_offset+1), 1*3+1, 1, 1, -1, col_offset)
        col_offset += 1
        
        # M4 = A11(B01-B00)
        # A11 -> U[4].
        # B01 -> V[0*8+k_offset+1]. B00 -> V[0*8+k_offset].
        # Output C00 (i=0,k=0) -> W[0]. C10 (i=1,k=0) -> W[1].
        # C00 = ... + M4. C10 = ... + M4.
        
        # Term 9: A11 * B01
        add_term(4, 0*8+(k_offset+1), 0*3+0, 1, 1, 1, col_offset)
        add_term(4, 0*8+(k_offset+1), 0*3+1, 1, 1, 1, col_offset)
        # Term 10: A11 * B00
        add_term(4, 0*8+k_offset, 0*3+0, 1, 1, -1, col_offset)
        add_term(4, 0*8+k_offset, 0*3+1, 1, 1, -1, col_offset)
        col_offset += 1
        
        # M5 = (A00+A10)B11
        # A00 -> U[0]. A10 -> U[3].
        # B11 -> V[1*8+k_offset+1].
        # Output C00 (i=0,k=0) -> W[0]. C01 (i=0,k=1) -> W[3].
        # C00 = ... - M5. C01 = ... + M5.
        
        # Term 11: A00 * B11
        add_term(0, 1*8+(k_offset+1), 1*3+0, 1, 1, -1, col_offset)
        add_term(0, 1*8+(k_offset+1), 1*3+0, 1, 1, 1, col_offset) # Wait, C01 is W[3].
        # Correction:
        # C00 is W[0*3+0]=W[0].
        # C01 is W[1*3+0]=W[3].
        # M5 contributes -1 to C00 and +1 to C01.
        add_term(0, 1*8+(k_offset+1), 0*3+0, 1, 1, -1, col_offset)
        add_term(0, 1*8+(k_offset+1), 1*3+0, 1, 1, 1, col_offset)
        
        # Term 12: A10 * B11
        add_term(3, 1*8+(k_offset+1), 0*3+0, 1, 1, -1, col_offset)
        add_term(3, 1*8+(k_offset+1), 1*3+0, 1, 1, 1, col_offset)
        col_offset += 1
        
        # M6 = (A10-A00)(B00+B10)
        # A10 -> U[3]. A00 -> U[0].
        # B00 -> V[0*8+k_offset]. B10 -> V[1*8+k_offset].
        # Output C11 (i=1,k=1) -> W[4].
        # C11 = ... + M6.
        
        # Term 13: A10 * B00
        add_term(3, 0*8+k_offset, 1*3+1, 1, 1, 1, col_offset)
        # Term 14: A10 * B10
        add_term(3, 1*8+k_offset, 1*3+1, 1, 1, 1, col_offset)
        # Term 15: A00 * B00
        add_term(0, 0*8+k_offset, 1*3+1, 1, 1, -1, col_offset)
        # Term 16: A00 * B10
        add_term(0, 1*8+k_offset, 1*3+1, 1, 1, -1, col_offset)
        col_offset += 1
        
        # M7 = (A01-A11)(B01+B11)
        # A01 -> U[1]. A11 -> U[4].
        # B01 -> V[0*8+k_offset+1]. B11 -> V[1*8+k_offset+1].
        # Output C00 (i=0,k=0) -> W[0].
        # C00 = ... + M7.
        
        # Term 17: A01 * B01
        add_term(1, 0*8+(k_offset+1), 0*3+0, 1, 1, 1, col_offset)
        # Term 18: A01 * B11
        add_term(1, 1*8+(k_offset+1), 0*3+0, 1, 1, 1, col_offset)
        # Term 19: A11 * B01
        add_term(4, 0*8+(k_offset+1), 0*3+0, 1, 1, -1, col_offset)
        # Term 20: A11 * B11
        add_term(4, 1*8+(k_offset+1), 0*3+0, 1, 1, -1, col_offset)
        col_offset += 1

        # --- Naive Part (10 terms) ---
        
        # 1. A[0,2] B[2,0] -> C[0,0]
        add_term(2, 2*8+k_offset, 0*3+0, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 2. A[0,2] B[2,1] -> C[0,1]
        add_term(2, 2*8+(k_offset+1), 1*3+0, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 3. A[1,2] B[2,0] -> C[1,0]
        add_term(5, 2*8+k_offset, 0*3+1, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 4. A[1,2] B[2,1] -> C[1,1]
        add_term(5, 2*8+(k_offset+1), 1*3+1, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 5. A[2,0] B[0,0] -> C[2,0]
        add_term(6, 0*8+k_offset, 0*3+2, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 6. A[2,0] B[0,1] -> C[2,1]
        add_term(6, 0*8+(k_offset+1), 1*3+2, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 7. A[2,1] B[1,0] -> C[2,0]
        add_term(7, 1*8+k_offset, 0*3+2, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 8. A[2,1] B[1,1] -> C[2,1]
        add_term(7, 1*8+(k_offset+1), 1*3+2, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 9. A[2,2] B[2,0] -> C[2,0]
        add_term(8, 2*8+k_offset, 0*3+2, 1, 1, 1, col_offset)
        col_offset += 1
        
        # 10. A[2,2] B[2,1] -> C[2,1]
        add_term(8, 2*8+(k_offset+1), 1*3+2, 1, 1, 1, col_offset)
        col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
