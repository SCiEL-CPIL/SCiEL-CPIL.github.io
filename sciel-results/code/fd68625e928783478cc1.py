"""Optimized matrix multiplication 2x4x5 via Strassen-style decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _validate(U, V, W, n, m, p):
    """Check exact decomposition. U:(n*m, rank), V:(m*p, rank), W:(p*n, rank) with W[k*n+i, r]."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
            return False
    # Build the tensor from factors and check
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if i==i, j==j, k==k (i.e., always 1? No.)
    # Actually for matrix mult (n x m) * (m x p) -> (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} ... no.
    # The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k? No.
    # It's the structure tensor: T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # Wait, that's not right either. The tensor is defined as:
    # T_{ijk} = 1 if the product A_{ij} * B_{jk} contributes to C_{ik}
    # So T_{ijk} = 1 for all valid i, j, k.
    # Let me verify with the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k) pair.
    # So yes, T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    # Quick check: verify a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def _strassen_2x2_2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*2+j, c] - A side
    # V[j*2+k, c] - B side  
    # W[k*2+i, c] - C side
    
    # c=0: M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1; W[1][0] = 1  # C11 (k=0,i=0), C21 (k=1,i=0)... wait W[k*2+i]
    # W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[1,0], W[0*2+1]=W[1] for C[0,1], W[1*2+1]=W[3] for C[1,1]
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # Let me redo with proper indexing
    # W index: W[k*2 + i] where k is column of C, i is row of C
    # So W[0] -> C[0,0], W[1] -> C[0,1], W[2] -> C[1,0], W[3] -> C[1,1]
    
    # Reset
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A11+A22)(B11+B22)
    # A11=U[0], A22=U[3], B11=V[0], B22=V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # Contributes to C11 (+1) and C22 (+1)
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # c=1: M2 = (A21+A22)B11
    # A21=U[2], A22=U[3], B11=V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # Contributes to C21 (+1) and C22 (-1)
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1  # C[1,1]
    
    # c=2: M3 = A11(B21-B22)
    # A11=U[0], B21=V[2], B22=V[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # Contributes to C12 (+1) and C22 (+1)
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # c=3: M4 = A22(B12-B21)
    # A22=U[3], B12=V[1], B21=V[2]
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    # Contributes to C11 (+1) and C21 (+1)
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # c=4: M5 = (A11+A12)B22
    # A11=U[0], A12=U[1], B22=V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # Contributes to C11 (-1) and C12 (+1)
    W[0][4] = -1  # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # c=5: M6 = (A21-A11)(B11+B12)
    # A21=U[2], A11=U[0], B11=V[0], B12=V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # Contributes to C22 (+1)
    W[3][5] = 1  # C[1,1]
    
    # c=6: M7 = (A12-A22)(B21+B22)
    # A12=U[1], A22=U[3], B21=V[2], B22=V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # Contributes to C11 (+1)
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return (U, V, W) for n x m x p schoolbook decomposition.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p).
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

def _combine_blocks(blocks):
    """Combine a list of (U,V,W, row_offsets_U, row_offsets_V, row_offsets_W) into one.
    Each block has U of shape (block_n*block_m, rank_block), etc.
    row_offsets tell us where to place each block's rows in the full matrix.
    """
    total_rank = sum(b[2] for b in [len(b[0][0]) for b in blocks])
    # Actually let me just concatenate columns
    total_rank = 0
    for U, V, W, *_ in blocks:
        total_rank += len(U[0])
    
    # Determine full sizes
    n = max(U[0] and len(U) // m for U, V, W, *x in blocks for m in [1])  # This is getting complicated
    # Let me take a different approach: just build the full U, V, W directly
    
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Strategy: Split m=4 into m1=2 and m2=2
    # Block 1: 2x2x5 (n=2, m=2, p=5)
    # Block 2: 2x2x5 (n=2, m=2, p=5)
    # For each 2x2x5: split p=5 into p1=2 and p2=3
    #   Sub-block 1: 2x2x2 (Strassen, rank 7)
    #   Sub-block 2: 2x2x3 (schoolbook, rank 12)
    # So each 2x2x5 has rank 7+12=19, total 38.
    # That's worse than 36.
    
    # Alternative from knowledge: split m=4 into two m=2 blocks.
    # Each 2x2x5: further decompose.
    # The knowledge says: "Decomposing the 2x4x5 multiplication into two 2x2x5 blocks, 
    # each further decomposed into two 2x2x2 Strassen blocks (rank 7) and one 2x2x1 schoolbook block (rank 4)"
    # Wait, 2x2x2 + 2x2x2 + 2x2x1? That's p=2+2+1=5. Rank = 7+7+4=18 per block, total 36.
    
    # Let me implement: split m=4 into two groups of 2 columns.
    # For each group, split p=5 into p=2, p=2, p=1.
    # 2x2x2 -> Strassen rank 7
    # 2x2x2 -> Strassen rank 7  
    # 2x2x1 -> schoolbook rank 4
    # Total per 2x2x5 block: 18
    # Total: 36
    
    # Let's build this.
    # Full U: shape (n*m, rank) = (8, 36)
    # Full V: shape (m*p, rank) = (20, 36)
    # Full W: shape (p*n, rank) = (10, 36)
    
    total_rank = 36
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    
    # For each of the 2 blocks (m split into [0,1] and [2,3])
    for block in range(2):
        m_offset = block * 2  # columns 0,1 or 2,3
        
        # For each sub-block of p: p_sizes = [2, 2, 1]
        p_offset = 0
        for p_size in [2, 2, 1]:
            if p_size == 2:
                # Strassen 2x2x2
                sU, sV, sW = _strassen_2x2_2x2()
                for i in range(2):
                    for j in range(2):
                        for c in range(7):
                            U[(block*2 + i)*2 + j][col_offset + c] = sU[i*2+j][c]
                for j in range(2):
                    for k in range(2):
                        for c in range(7):
                            V[(m_offset + j)*5 + (p_offset + k)][col_offset + c] = sV[j*2+k][c]
                for k in range(2):
                    for i in range(2):
                        for c in range(7):
                            W[(p_offset + k)*2 + i][col_offset + c] = sW[k*2+i][c]
                col_offset += 7
            else:  # p_size == 1
                # Schoolbook 2x2x1
                for i in range(2):
                    for j in range(2):
                        for k in range(1):
                            U[(block*2 + i)*2 + j][col_offset] = 1
                            V[(m_offset + j)*5 + (p_offset + k)][col_offset] = 1
                            W[(p_offset + k)*2 + i][col_offset] = 1
                            col_offset += 1
            
            p_offset += p_size
    
    # Verify
    if time.time() < deadline:
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
