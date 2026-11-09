"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = 2
    rank = 7
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each rank-1 term is (U_col, V_col, W_col) where:
    # U[i*2+j] has value for A_ij, V[j*2+k] has value for B_jk, W[k*2+i] has value for C_ki
    
    # Let's define each of the 7 terms.
    # Term 1: M1 = (A11+A22)*(B11+B22)
    #   A11 = U[0], A22 = U[3], B11 = V[0], B22 = V[3]
    #   C11 += M1, C22 += M1
    #   W[0*2+0]=W[0] for C11 (k=0,i=0), W[1*2+1]=W[3] for C22 (k=1,i=1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A21+A22)*B11
    #   A21 = U[2], A22 = U[3], B11 = V[0]
    #   C21 += M2 (k=0,i=1 -> W[0*2+1]=W[1])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Term 3: M3 = A11*(B12-B22)
    #   A11 = U[0], B12 = V[1], B22 = V[3]
    #   C12 += M3 (k=1,i=0 -> W[1*2+0]=W[2]), C22 += M3 (k=1,i=1 -> W[3])
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A22*(B21-B11)
    #   A22 = U[3], B21 = V[2], B11 = V[0]
    #   C11 += M4 (k=0,i=0 -> W[0]), C21 += M4 (k=0,i=1 -> W[1])
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A11+A12)*B22
    #   A11 = U[0], A12 = U[1], B22 = V[3]
    #   C11 -= M5 (k=0,i=0 -> W[0]), C12 += M5 (k=1,i=0 -> W[2])
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A21-A11)*(B11+B12)
    #   A21 = U[2], A11 = U[0], B11 = V[0], B12 = V[1]
    #   C22 += M6 (k=1,i=1 -> W[3])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A12-A22)*(B21+B22)
    #   A12 = U[1], A22 = U[3], B21 = V[2], B22 = V[3]
    #   C11 += M7 (k=0,i=0 -> W[0]), C22 += M7 (k=1,i=1 -> W[3])
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Return (U, V, W) for n x m * m x p schoolbook (rank n*m*p).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks, n, m, p, row_starts, col_starts):
    """
    Combine independent block decompositions.
    blocks: list of (U_b, V_b, W_b, rank_b)
    row_starts, col_starts: for mapping local indices to global
    """
    total_rank = sum(b[3] for b in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for idx, (Ub, Vb, Wb, rb) in enumerate(blocks):
        for i in range(n):
            for j in range(m):
                for c in range(rb):
                    U[i*m+j][offset+c] = Ub[i*m+j][c]
        for j in range(m):
            for k in range(p):
                for c in range(rb):
                    V[j*p+k][offset+c] = Vb[j*p+k][c]
        for k in range(p):
            for i in range(n):
                for c in range(rb):
                    W[k*n+i][offset+c] = Wb[k*n+i][c]
        offset += rb
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition 5 into 2+2+1
    # Rows: 0,1 | 2,3 | 4
    # Cols of A (rows of B): 0,1 | 2,3 | 4
    # Cols of B (rows of C): 0,1 | 2,3 | 4
    
    # 9 blocks: 3x3 grid
    # Block (r, c) where r is row-block index, c is col-block index
    # Sizes:
    #   row_sizes = [2, 2, 1]
    #   col_sizes = [2, 2, 1]  (this is the "m" dimension)
    #   out_sizes = [2, 2, 1]  (this is the "p" dimension, same as col_sizes for square)
    
    row_sizes = [2, 2, 1]
    col_sizes = [2, 2, 1]  # m dimension
    out_sizes = [2, 2, 1]  # p dimension
    
    # For each block (ri, ci, oi), we need to compute:
    # C[ri block, oi block] += sum over mi blocks: A[ri, mi] * B[mi, oi]
    # But in our tensor decomposition, each block (ri, mi, oi) contributes independently.
    # Block (ri, mi, oi) has size: n_ri x m_mi * m_mi x p_oi -> rank depends on algorithm.
    
    # For 2x2x2 blocks, use Strassen (rank 7)
    # For other sizes, use schoolbook
    
    blocks = []
    
    for ri in range(3):
        for mi in range(3):
            for oi in range(3):
                nr = row_sizes[ri]
                nm = col_sizes[mi]
                np_ = out_sizes[oi]
                
                if nr == 2 and nm == 2 and np_ == 2:
                    Ub, Vb, Wb = _strassen_2x2x2()
                    rb = 7
                else:
                    Ub, Vb, Wb = _schoolbook_block(nr, nm, np_)
                    rb = nr * nm * np_
                
                blocks.append((Ub, Vb, Wb, rb, ri, mi, oi))
    
    total_rank = sum(b[3] for b in blocks)
    
    # Build global U, V, W
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for (Ub, Vb, Wb, rb, ri, mi, oi) in blocks:
        nr = row_sizes[ri]
        nm = col_sizes[mi]
        np_ = out_sizes[oi]
        
        # Row offset in U: sum of row_sizes[0..ri-1]
        row_off = sum(row_sizes[:ri])
        # Col offset in U (m dimension): sum of col_sizes[0..mi-1]
        m_off = sum(col_sizes[:mi])
        # Row offset in V (m dimension): m_off
        # Col offset in V (p dimension): sum of out_sizes[0..oi-1]
        p_off = sum(out_sizes[:oi])
        # Row offset in W (p dimension): p_off
        # Col offset in W (n dimension): row_off
        
        for i in range(nr):
            for j in range(nm):
                for c in range(rb):
                    U[(row_off+i)*m + (m_off+j)][offset+c] = Ub[i*nm+j][c]
        
        for j in range(nm):
            for k in range(np_):
                for c in range(rb):
                    V[(m_off+j)*p + (p_off+k)][offset+c] = Vb[j*np_+k][c]
        
        for k in range(np_):
            for i in range(nr):
                for c in range(rb):
                    W[(p_off+k)*n + (row_off+i)][offset+c] = Wb[k*nr+i][c]
        
        offset += rb
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
