"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _schoolbook_decomp(n, m, p):
    """Return (U, V, W) for n×m · m×p → n×p with rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2():
    """Strassen 2×2×2 decomposition, rank 7."""
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # U[i*m+j, c] for i,j in {0,1}, m=2
    # V[j*p+k, c] for j,k in {0,1}, p=2
    # W[k*n+i, c] for k,i in {0,1}, n=2
    #
    # U rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # V rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # W rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3  [k*n+i]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22) -> C contribution
    # A11 = U[0], A22 = U[3]; B11 = V[0], B22 = V[3]
    # W: k*n+i; M1 contributes to C11 (i=0,k=0), C12(i=0,k=1), C21(i=1,k=0), C22(i=1,k=1)
    # C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    # M1: C11+=1, C12+=0, C21+=0, C22+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11: k=0,i=0 -> W[0]; C22: k=1,i=1 -> W[3]
    
    # M2: (A21+A22)B11
    # A21=U[2], A22=U[3]; B11=V[0]
    # C21 += M2, C22 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1; W[3][1] = -1  # C21: k=0,i=1 -> W[1*2+0]=W[2]? 
    # Wait, W[k*n+i], n=2. C21 means i=1 (row 2 of C), k=0 (col 1 of C, 0-indexed)
    # So W[k*n+i] = W[0*2+1] = W[1] for C21
    # C22: i=1, k=1 -> W[1*2+1] = W[3]
    # Let me redo W indexing: W index = k*n + i
    # C11: i=0,k=0 -> W[0]; C12: i=0,k=1 -> W[1]; C21: i=1,k=0 -> W[2]; C22: i=1,k=1 -> W[3]
    
    # Redo M1:
    W[0][0] = 1; W[3][0] = 1  # C11 and C22
    
    # M2: C21+=1, C22-=1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A11(B21-B22)
    # A11=U[0]; B21=V[2], B22=V[3]
    # C12 += M3, C22 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22(B12-B11)
    # A22=U[3]; B12=V[1], B11=V[0]
    # C11 += M4, C21 += M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11+A12)B22
    # A11=U[0], A12=U[1]; B22=V[3]
    # C11 -= M5, C12 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)(B11+B12)
    # A21=U[2], A11=U[0]; B11=V[0], B12=V[1]
    # C22 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # A12=U[1], A22=U[3]; B21=V[2], B22=V[3]
    # C11 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _strassen_3x3():
    """3×3×3 decomposition using 2+1 split with Strassen on 2×2 parts. Rank = 7+6+6+6+26/4... """
    # Actually let me just use schoolbook for 3×3 to be safe, rank=27
    # Or use a better construction.
    # For now, use schoolbook rank 27.
    return _schoolbook_decomp(3, 3, 3)

def _rect_2_2_3():
    """(2×2)·(2×3)→(2×3), rank 12 schoolbook."""
    return _schoolbook_decomp(2, 2, 3)

def _rect_2_3_2():
    """(2×3)·(3×2)→(2×2), rank 12 schoolbook."""
    return _schoolbook_decomp(2, 3, 2)

def _rect_2_3_3():
    """(2×3)·(3×3)→(2×3), rank 18 schoolbook."""
    return _schoolbook_decomp(2, 3, 3)

def _rect_3_2_2():
    """(3×2)·(2×2)→(3×2), rank 12 schoolbook."""
    return _schoolbook_decomp(3, 2, 2)

def _rect_3_2_3():
    """(3×2)·(2×3)→(3×3), rank 18 schoolbook."""
    return _schoolbook_decomp(3, 2, 3)

def _rect_3_3_2():
    """(3×3)·(3×2)→(3×2), rank 18 schoolbook."""
    return _schoolbook_decomp(3, 3, 2)

def _combine_blocks(U1, V1, W1, r1, U2, V2, W2, r2):
    """Concatenate two decompositions."""
    rank = r1 + r2
    U = [row + [0]*r2 for row in U1] + [[0]*r1 + row for row in U2]
    V = [row + [0]*r2 for row in V1] + [[0]*r1 + row for row in V2]
    W = [row + [0]*r2 for row in W1] + [[0]*r1 + row for row in W2]
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 5 = 2 + 3 block split
    # Blocks: (2,2,2), (2,2,3), (2,3,2), (2,3,3), (3,2,2), (3,2,3), (3,3,2), (3,3,3)
    # 
    # U shape: (5*5, rank) = (25, rank)
    # V shape: (5*5, rank) = (25, rank)  
    # W shape: (5*5, rank) = (25, rank)
    #
    # U row index: i*m + j where i is row of A, j is col of A
    # For block (ni, mj, pk) where A is split as (n1+n2)×(m1+m2), B as (m1+m2)×(p1+p2), C as (n1+n2)×(p1+p2)
    # Block (ni, mj, pk): A block is ni×mj, B block is mj×pk, C block is ni×pk
    # U rows for this block: i in [ni_block_start, ni_block_end), j in [mj_block_start, mj_block_end)
    #   U_local row = i_local * mj + j_local
    #   U_global row = i_global * 5 + j_global
    # V rows: j in [mj_block_start, mj_block_end), k in [pk_block_start, pk_block_end)
    #   V_local row = j_local * pk + k_local
    #   V_global row = j_global * 5 + k_global
    # W rows: k in [pk_block_start, pk_block_end), i in [ni_block_start, ni_block_end)
    #   W_local row = k_local * ni + i_local
    #   W_global row = k_global * 5 + i_global
    
    # Define blocks with their (ni, mj, pk) and offsets
    # A: rows 0-4, cols 0-4. Split rows: [0-1] (size 2), [2-4] (size 3)
    #     cols:   [0-1] (size 2), [2-4] (size 3)
    # B: rows 0-4 (A's cols), cols 0-4. Same split.
    # C: rows 0-4 (A's rows), cols 0-4 (B's cols). Same split.
    
    blocks = [
        # (ni_size, mj_size, pk_size, ni_offset, mj_offset, pk_offset, decomp_func)
        (2, 2, 2, 0, 0, 0, _strassen_2x2),
        (2, 2, 3, 0, 0, 2, _rect_2_2_3),
        (2, 3, 2, 0, 2, 0, _rect_2_3_2),
        (2, 3, 3, 0, 2, 2, _rect_2_3_3),
        (3, 2, 2, 2, 0, 0, _rect_3_2_2),
        (3, 2, 3, 2, 0, 2, _rect_3_2_3),
        (3, 3, 2, 2, 2, 0, _rect_3_3_2),
        (3, 3, 3, 2, 2, 2, _strassen_3x3),
    ]
    
    # Compute total rank
    total_rank = 0
    block_decomps = []
    for ni, mj, pk, noi, mjo, pko, func in blocks:
        U_local, V_local, W_local, r = func()
        block_decomps.append((ni, mj, pk, noi, mjo, pko, U_local, V_local, W_local, r))
        total_rank += r
    
    # Allocate global matrices
    U_global = [[0] * total_rank for _ in range(25)]
    V_global = [[0] * total_rank for _ in range(25)]
    W_global = [[0] * total_rank for _ in range(25)]
    
    col_offset = 0
    for ni, mj, pk, noi, mjo, pko, U_local, V_local, W_local, r in block_decomps:
        # Place U_local into U_global
        for i_local in range(ni):
            for j_local in range(mj):
                i_global = noi + i_local
                j_global = mjo + j_local
                u_row = i_global * 5 + j_global
                u_local_row = i_local * mj + j_local
                for c in range(r):
                    U_global[u_row][col_offset + c] = U_local[u_local_row][c]
        
        # Place V_local into V_global
        for j_local in range(mj):
            for k_local in range(pk):
                j_global = mjo + j_local
                k_global = pko + k_local
                v_row = j_global * 5 + k_global
                v_local_row = j_local * pk + k_local
                for c in range(r):
                    V_global[v_row][col_offset + c] = V_local[v_local_row][c]
        
        # Place W_local into W_global
        for k_local in range(pk):
            for i_local in range(ni):
                k_global = pko + k_local
                i_global = noi + i_local
                w_row = k_global * 5 + i_global
                w_local_row = k_local * ni + i_local
                for c in range(r):
                    W_global[w_row][col_offset + c] = W_local[w_local_row][c]
        
        col_offset += r
    
    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
