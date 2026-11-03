"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 135  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strategy: Use the Kronecker product structure to build a rank-49 decomposition
    # for 4x4x4 from the rank-7 Strassen decomposition for 2x2x2.
    # Then split the 4x8 problem into two 4x4x4 problems (left and right halves of B),
    # sharing the U factor (since A is the same) to get total rank 49 + 49 = 98.
    
    # Actually, let's reconsider. The tensor for 4x4 by 4x8 is:
    # T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..7
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Split k into k' in 0..3 (left) and k'+4 in 0..3 (right).
    # Left: T[i,j,k'] for k' in 0..3 is a 4x4x4 tensor.
    # Right: T[i,j,k'+4] for k' in 0..3 is also a 4x4x4 tensor.
    
    # If we have a rank-r decomposition for 4x4x4:
    # T4[i,j,k'] = sum_c U4[i,j,c] * V4[j,k',c] * W4[k',i,c]
    
    # Then for the full problem:
    # U[i*m+j, c] = U4[i,j,c] for c in 0..r-1 (shared for both left and right)
    # V[j*p+k, c] = V4[j,k,c] for k in 0..3, c in 0..r-1 (left block)
    # V[j*p+k, c+r] = V4[j,k-4,c] for k in 4..7 (right block)
    # W[k*n+i, c] = W4[k,i,c] for k in 0..3, c in 0..r-1 (left block)
    # W[k*n+i, c+r] = W4[k-4,i,c] for k in 4..7 (right block)
    
    # Total rank = 2r.
    
    # Now I need the rank-49 decomposition for 4x4x4.
    # Using the Kronecker product of the 2x2x2 Strassen decomposition with itself.
    
    # 2x2x2 Strassen decomposition (rank 7):
    # T2[i,j,k] = sum_{a=0}^{6} u2[i,a] * v2[j,a] * w2[k,a]
    u2 = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0: A11
        [0, 0, 0, 0, 1, 0, 1],   # i=1: A12
        [0, 1, 0, 0, 0, 1, 0],   # i=2: A21
        [1, 1, 0, 1, 0, 0, -1],  # i=3: A22
    ]
    v2 = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0: B11
        [0, 0, 1, 0, 0, 1, 1],   # j=1: B12
        [0, 0, 0, 1, 0, 0, 1],   # j=2: B21
        [1, 0, -1, 0, 1, 0, 0],  # j=3: B22
    ]
    w2 = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0: C11
        [0, 0, 1, 0, 1, 0, 0],   # k=1: C12
        [0, 1, 0, 1, 0, 0, 0],   # k=2: C21
        [1, -1, 1, 0, 0, 1, 0],  # k=3: C22
    ]
    
    # Verify: for 2x2x2, T2[i,j,k] should be 1 if (i,j,k) corresponds to a valid multiplication term.
    # Actually, the tensor is T2[i,j,k] = 1 for all i,j,k in 0..3 (since it's the full multiplication tensor).
    # Wait, no. The tensor for 2x2 by 2x2 multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k in 0..3 (2x2x2 = 8 elements? No, 2x2x2 has 8 elements).
    # Actually, i in 0..1, j in 0..1, k in 0..1 for 2x2x2.
    # But the indices above are 0..3, which suggests they're using a flattened 2x2 block index.
    
    # Let me reconsider. The standard Strassen decomposition for 2x2x2 uses:
    # i, j, k in {0, 1} for the 2x2 matrix indices.
    # But the vectors u2, v2, w2 above have 4 rows each, which is confusing.
    
    # I think the issue is that the "2x2x2" here refers to the block structure, not the scalar indices.
    # Let me use a cleaner approach.
    
    # For 2x2 by 2x2 matrix multiplication:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # The tensor T[i,j,k] where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # T[i,j,k] = 1 for all i,j,k in {0,1}.
    #
    # Strassen decomposition:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # So the rank-7 decomposition has:
    # U (for A): 2x2 matrix -> 4 entries, 7 columns
    # V (for B): 2x2 matrix -> 4 entries, 7 columns  
    # W (for C): 2x2 matrix -> 4 entries, 7 columns
    
    # U[i*2+j, c] for i,j in {0,1}, c in 0..6
    # V[j*2+k, c] for j,k in {0,1}, c in 0..6
    # W[k*2+i, c] for k,i in {0,1}, c in 0..6
    
    # Let me define these properly:
    # Row index for U: i*2+j where i is row, j is col of A
    # Row index for V: j*2+k where j is row of B (= col of A), k is col of B
    # Row index for W: k*2+i where k is col of C (= col of B), i is row of C (= row of A)
    
    U_2x2 = [
        [1, 0, 1, 0, 1, -1, 0],  # (i=0,j=0): a00 in M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],   # (i=0,j=1): a01 in M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # (i=1,j=0): a10 in M2, M6
        [1, 1, 0, 1, 0, 0, -1],  # (i=1,j=1): a11 in M1, M2, M4, M7
    ]
    
    V_2x2 = [
        [1, 1, 0, -1, 0, 1, 0],  # (j=0,k=0): b00 in M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 1],   # (j=0,k=1): b01 in M3, M6, M7
        [0, 0, 0, 1, 0, 0, 1],   # (j=1,k=0): b10 in M4, M7
        [1, 0, -1, 0, 1, 0, 0],  # (j=1,k=1): b11 in M1, M3, M5
    ]
    
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # (k=0,i=0): c00 in M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # (k=1,i=0): c01 in M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # (k=0,i=1): c10 in M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # (k=1,i=1): c11 in M1, M2, M3, M6
    ]
    
    # Now for 4x4x4, we can use the Kronecker product structure.
    # Write i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2 where i1,j1,k1 in {0,1} (block indices)
    # and i2,j2,k2 in {0,1} (local indices within block).
    
    # The 4x4x4 tensor can be decomposed as:
    # T4[i,j,k] = sum_{a,b=0}^{6} U4[i,j,a*7+b] * V4[j,k,a*7+b] * W4[k,i,a*7+b]
    
    # Where:
    # U4[i1*2+i2, j1*2+j2, a*7+b] = U_2x2[i1*2+j1, a] * U_2x2[i2*2+j2, b]
    # V4[j1*2+j2, k1*2+k2, a*7+b] = V_2x2[j1*2+k1, a] * V_2x2[j2*2+k2, b]
    # W4[k1*2+k2, i1*2+i2, a*7+b] = W_2x2[k1*2+i1, a] * W_2x2[k2*2+i2, b]
    
    # This gives rank 49 for 4x4x4.
    
    # Now build the full 4x4x8 decomposition:
    r = 49  # rank of 4x4x4
    total_rank = 2 * r  # 98 for 4x4x8
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 98
    
    # Build U4, V4, W4 for 4x4x4 first
    U4 = [[0] * r for _ in range(16)]
    V4 = [[0] * r for _ in range(16)]
    W4 = [[0] * r for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    for a in range(7):
                        for b in range(7):
                            c = a * 7 + b
                            U4[i * 4 + j][c] = U_2x2[i1 * 2 + j1][a] * U_2x2[i2 * 2 + j2][b]
                            V4[j * 4 + (k1 := 0) * 2 + 0][c] = 0  # placeholder, will fill properly below
    
    # Actually, let me fill V4 and W4 properly:
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    for a in range(7):
                        for b in range(7):
                            c = a * 7 + b
                            V4[j * 4 + k][c] = V_2x2[j1 * 2 + k1][a] * V_2x2[j2 * 2 + k2][b]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    for a in range(7):
                        for b in range(7):
                            c = a * 7 + b
                            W4[k * 4 + i][c] = W_2x2[k1 * 2 + i1][a] * W_2x2[k2 * 2 + i2][b]
    
    # Now assemble the full 4x4x8 decomposition:
    # U[i*4+j, c] = U4[i*4+j, c] for c in 0..48 (shared)
    # V[j*8+k, c] = V4[j*4+k, c] for k in 0..3, c in 0..48 (left)
    # V[j*8+k, c+49] = V4[j*4+(k-4), c] for k in 4..7 (right)
    # W[k*4+i, c] = W4[k*4+i, c] for k in 0..3, c in 0..48 (left)
    # W[k*4+i, c+49] = W4[(k-4)*4+i, c] for k in 4..7 (right)
    
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i * 4 + j][c] = U4[i * 4 + j][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                # Left block
                V[j * 8 + k][c] = V4[j * 4 + k][c]
                # Right block
                V[j * 8 + (k + 4)][c + 49] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                # Left block
                W[k * 4 + i][c] = W4[k * 4 + i][c]
                # Right block
                W[(k + 4) * 4 + i][c + 49] = W4[k * 4 + i][c]
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback to schoolbook if we're running out of time
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
