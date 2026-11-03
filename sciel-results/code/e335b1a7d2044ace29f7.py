"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition dimensions into blocks
    # i (3) = 2 + 1
    # j (5) = 2 + 2 + 1
    # k (6) = 2 + 2 + 2
    i_blocks = [2, 1]
    j_blocks = [2, 2, 1]
    k_blocks = [2, 2, 2]
    
    def strassen_decompose():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U shape (4, 7): U[i*2+j] for A[i][j]
        V shape (4, 7): V[j*2+k] for B[j][k]
        W shape (4, 7): W[k*2+i] for C[i][k]
        W[0]=C11, W[2]=C12, W[1]=C21, W[3]=C22
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A: idx 0=A11, 3=A22; B: idx 0=B11, 3=B22
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += M1 (W[0]), C22 += M1 (W[3])
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        # A: idx 2=A21, 3=A22; B: idx 0=B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2 (W[1]), C22 -= M2 (W[3])
        W[1][1] = 1; W[3][1] = -1
        
        # M3 = A11(B21-B22)
        # A: idx 0=A11; B: idx 2=B21, 3=B22
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        # C12 += M3 (W[2]), C22 += M3 (W[3])
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = A22(B12-B11)
        # A: idx 3=A22; B: idx 1=B12, 0=B11
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        # C11 += M4 (W[0]), C21 += M4 (W[1])
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (A11+A12)B22
        # A: idx 0=A11, 1=A12; B: idx 3=B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 -= M5 (W[0]), C12 += M5 (W[2])
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A: idx 2=A21, 0=A11; B: idx 0=B11, 1=B12
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6 (W[3])
        W[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A: idx 1=A12, 3=A22; B: idx 2=B21, 3=B22
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += M7 (W[0])
        W[0][6] = 1
        
        return U, V, W
    
    def schoolbook_decompose(sa, sb, sc):
        """
        Schoolbook for sa x sb x sc.
        U shape (sa*sb, sa*sb*sc)
        V shape (sb*sc, sa*sb*sc)
        W shape (sc*sa, sa*sb*sc)
        """
        rank = sa * sb * sc
        U = [[0]*rank for _ in range(sa*sb)]
        V = [[0]*rank for _ in range(sb*sc)]
        W = [[0]*rank for _ in range(sc*sa)]
        col = 0
        for i in range(sa):
            for j in range(sb):
                for k in range(sc):
                    U[i*sb+j][col] = 1
                    V[j*sc+k][col] = 1
                    W[k*sa+i][col] = 1
                    col += 1
        return U, V, W
    
    # Get Strassen decomposition
    strassen_U, strassen_V, strassen_W = strassen_decompose()
    
    # Cache schoolbook decompositions
    schoolbook_cache = {}
    def get_schoolbook(sa, sb, sc):
        key = (sa, sb, sc)
        if key not in schoolbook_cache:
            schoolbook_cache[key] = schoolbook_decompose(sa, sb, sc)
        return schoolbook_cache[key]
    
    def get_block_decomp(sa, sb, sc):
        if sa == 2 and sb == 2 and sc == 2:
            return strassen_U, strassen_V, strassen_W, 7
        else:
            U, V, W = get_schoolbook(sa, sb, sc)
            return U, V, W, sa*sb*sc
    
    # Compute block offsets
    i_offsets = []
    off = 0
    for s in i_blocks:
        i_offsets.append(off)
        off += s
    
    j_offsets = []
    off = 0
    for s in j_blocks:
        j_offsets.append(off)
        off += s
    
    k_offsets = []
    off = 0
    for s in k_blocks:
        k_offsets.append(off)
        off += s
    
    # First pass: compute total rank
    total_rank = 0
    block_info = []
    for i_idx, (i_off, sa) in enumerate(zip(i_offsets, i_blocks)):
        for j_idx, (j_off, sb) in enumerate(zip(j_offsets, j_blocks)):
            for k_idx, (k_off, sc) in enumerate(zip(k_offsets, k_blocks)):
                _, _, _, rank = get_block_decomp(sa, sb, sc)
                total_rank += rank
                block_info.append((i_idx, j_idx, k_idx, sa, sb, sc, rank))
    
    # Allocate matrices
    U_total = [[0]*total_rank for _ in range(n*m)]
    V_total = [[0]*total_rank for _ in range(m*p)]
    W_total = [[0]*total_rank for _ in range(p*n)]
    
    # Fill in the decomposition
    col_offset = 0
    for (i_idx, j_idx, k_idx, sa, sb, sc, rank) in block_info:
        U_block, V_block, W_block, _ = get_block_decomp(sa, sb, sc)
        
        i_off = i_offsets[i_idx]
        j_off = j_offsets[j_idx]
        k_off = k_offsets[k_idx]
        
        for r in range(rank):
            for ui in range(sa*sb):
                val = U_block[ui][r]
                if val != 0:
                    U_total[i_off*sb + ui][col_offset + r] += val
            for vj in range(sb*sc):
                val = V_block[vj][r]
                if val != 0:
                    V_total[j_off*sc + vj][col_offset + r] += val
            for wk in range(sc*sa):
                val = W_block[wk][r]
                if val != 0:
                    W_total[k_off*sa + wk][col_offset + r] += val
        
        col_offset += rank
    
    if time.time() - start_time > DEADLINE:
        pass
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
