"""Matrix multiplication 3,5,6 with block Strassen decomposition."""
DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7 multiplications for 2x2 matrix multiply
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # The 7 rank-1 terms correspond to:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: 4x7, V: 4x7, W: 4x7
    # U rows: (i,j) for i in 0..1, j in 0..1 -> index i*2+j
    # V rows: (j,k) for j in 0..1, k in 0..1 -> index j*2+k
    # W rows: (k,i) for k in 0..1, i in 0..1 -> index k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]+=1, C11: W[3,0]+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: W[2,1]+=1, C11: W[3,1]+=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    # A00: U[0,2]=1
    # B10: V[2,2]=1, B11: V[3,2]=-1
    # C01: W[1,2]+=1, C11: W[3,2]+=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    # A11: U[3,3]=1
    # B01: V[1,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]+=1, C10: W[2,3]+=1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C00: W[0,4]+=-1, C01: W[1,4]+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B10: V[2,5]=1
    # C11: W[3,5]+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B01: V[1,6]=1, B11: V[3,6]=1
    # C00: W[0,6]+=1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(a, b, c):
    """Return U, V, W for a x b x c schoolbook decomposition."""
    rank = a * b * c
    U = [[0] * rank for _ in range(a * b)]
    V = [[0] * rank for _ in range(b * c)]
    W = [[0] * rank for _ in range(c * a)]
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i * b + j][col] = 1
                V[j * c + k][col] = 1
                W[k * a + i][col] = 1
                col += 1
    return U, V, W

def _block_rank(a, b, c):
    """Get optimal rank for a x b x c block."""
    if a == 2 and b == 2 and c == 2:
        return 7
    # For 2x2x4: two independent 2x2x2 Strassen
    if a == 2 and b == 2 and c == 4:
        return 14
    return a * b * c

def _decompose_block(a, b, c):
    """Decompose a x b x c block optimally."""
    if a == 2 and b == 2 and c == 2:
        return _strassen_2x2x2()
    if a == 2 and b == 2 and c == 4:
        # Two independent 2x2x2 Strassen blocks
        U1, V1, W1 = _strassen_2x2x2()
        U2, V2, W2 = _strassen_2x2x2()
        U = [row[:] for row in U1] + [row[:] for row in U2]
        V = [row[:] for row in V1] + [row[:] for row in V2]
        W = [row[:] for row in W1] + [row[:] for row in W2]
        return U, V, W
    return _schoolbook_block(a, b, c)

def _combine_blocks(block_list, n, m, p):
    """Combine block decompositions into full 3x5x6 decomposition.
    block_list: list of (a, b, c, U, V, W, i_off, j_off, k_off)
    """
    total_rank = sum(len(U[0]) for (_, _, _, U, _, _, _, _, _) in block_list)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (a, b, c, U, V, W, i_off, j_off, k_off) in block_list:
        block_rank = len(U[0])
        for i in range(a):
            for j in range(b):
                for c_idx in range(block_rank):
                    val_u = U[i * b + j][c_idx]
                    if val_u != 0:
                        U_full[(i_off + i) * m + (j_off + j)][col_offset + c_idx] = val_u
        for j in range(b):
            for k in range(c):
                for c_idx in range(block_rank):
                    val_v = V[j * c + k][c_idx]
                    if val_v != 0:
                        V_full[(j_off + j) * p + (k_off + k)][col_offset + c_idx] = val_v
        for k in range(c):
            for i in range(a):
                for c_idx in range(block_rank):
                    val_w = W[k * a + i][c_idx]
                    if val_w != 0:
                        W_full[(k_off + k) * n + (i_off + i)][col_offset + c_idx] = val_w
        col_offset += block_rank
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication: T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # Wait, no. The matrix multiplication tensor T_{ijk} is defined such that
    # C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block decomposition: 3=1+2, 5=2+3, 6=2+4
    # Blocks: (a,b,c) with offsets (i_off, j_off, k_off)
    blocks_spec = [
        (1, 2, 2, 0, 0, 0),
        (1, 2, 4, 0, 0, 2),
        (1, 3, 2, 0, 2, 0),
        (1, 3, 4, 0, 2, 2),
        (2, 2, 2, 1, 0, 0),
        (2, 2, 4, 1, 0, 2),
        (2, 3, 2, 1, 2, 0),
        (2, 3, 4, 1, 2, 2),
    ]
    
    block_list = []
    for (a, b, c, i_off, j_off, k_off) in blocks_spec:
        U, V, W = _decompose_block(a, b, c)
        block_list.append((a, b, c, U, V, W, i_off, j_off, k_off))
    
    U_full, V_full, W_full = _combine_blocks(block_list, n, m, p)
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
