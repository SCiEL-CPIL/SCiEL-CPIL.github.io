"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) corresponding to rank-7 decomposition.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Indexing: A[i,j] where i,j in {0,1}
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient of C[i,k] in product c
    
    # Product 0: M1 = (a11+a22)(b11+b22)
    # A coefficients: a11=1, a22=1, others 0
    # B coefficients: b11=1, b22=1, others 0
    # C coefficients: C11+=1, C22+=1
    
    U = [[0.0]*7 for _ in range(4)]  # A part: indices 0=a11, 1=a12, 2=a21, 3=a22
    V = [[0.0]*7 for _ in range(4)]  # B part: indices 0=b11, 1=b12, 2=b21, 3=b22
    W = [[0.0]*7 for _ in range(4)]  # C part: indices 0=C11, 1=C12, 2=C21, 3=C22
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0  # a11
    U[3][0] = 1.0  # a22
    V[0][0] = 1.0  # b11
    V[3][0] = 1.0  # b22
    W[0][0] = 1.0  # C11
    W[3][0] = 1.0  # C22
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0  # a21
    U[3][1] = 1.0  # a22
    V[0][1] = 1.0  # b11
    W[2][1] = 1.0  # C21
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0  # b12
    V[3][2] = -1.0 # b22
    W[1][2] = 1.0  # C12
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0  # b21
    V[0][3] = -1.0 # b11
    W[0][3] = 1.0  # C11
    W[2][3] = 1.0  # C21
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0  # a11
    U[1][4] = 1.0  # a12
    V[3][4] = 1.0  # b22
    W[0][4] = -1.0 # C11
    W[1][4] = 1.0  # C12
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0  # a21
    U[0][5] = -1.0 # a11
    V[0][5] = 1.0  # b11
    V[1][5] = 1.0  # b12
    W[3][5] = 1.0  # C22
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0  # a12
    U[3][6] = -1.0 # a22
    V[2][6] = 1.0  # b21
    V[3][6] = 1.0  # b22
    W[0][6] = 1.0  # C11
    W[3][6] = 1.0  # C22
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Return schoolbook decomposition for n_block x m_block * m_block x p_block.
    Returns U, V, W each of shape (n_block*m_block, n_block*m_block*p_block) etc.
    """
    rank = n_block * m_block * p_block
    U = [[0.0]*rank for _ in range(n_block*m_block)]
    V = [[0.0]*rank for _ in range(m_block*p_block)]
    W = [[0.0]*rank for _ in range(p_block*n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i*m_block+j][col] = 1.0
                V[j*p_block+k][col] = 1.0
                W[k*n_block+i][col] = 1.0
                col += 1
    return U, V, W

def _embed_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, n_total, m_total, p_total):
    """Embed a block decomposition into the full matrix.
    U_block: (n_b*m_b, rank) -> U_full: (n_total*m_total, rank)
    V_block: (m_b*p_b, rank) -> V_full: (m_total*p_total, rank)
    W_block: (p_b*n_b, rank) -> W_full: (p_total*n_total, rank)
    """
    n_b = len(U_block) // m_block if m_total > 0 else 0
    # Infer block sizes from offsets and block shapes
    n_b = len(U_block) // (m_block if m_block else 1)
    # Actually, let's pass block sizes explicitly
    pass

def construct():
    """
    Find rank-minimal tensor decomposition for matrix multiplication (3x5) * (5x5).
    
    Uses block-structured decomposition:
    - Split the 5-dimensional middle space (m) into 2+3
    - Split the 5-dimensional output space (p) into 2+3  
    - Apply Strassen to the 2x2x2 sub-block (saves 1 rank: 8->7)
    - Use schoolbook for all other blocks
    
    Total rank: 
    - 2x2x2 block: 7 (Strassen)
    - 2x3x2 block: 12 (schoolbook)
    - 3x2x2 block: 12 (schoolbook)
    - 2x3x3 block: 18 (schoolbook)
    - 3x2x3 block: 18 (schoolbook)
    - 3x3x2 block: 18 (schoolbook)
    - 3x3x3 block: 27 (schoolbook)
    Total: 7+12+12+18+18+18+27 = 112? That's worse!
    
    Wait, let me reconsider. The blocks are:
    A is 3x5, B is 5x5, C is 3x5
    Split m (middle) into m1=2, m2=3
    Split p (output cols) into p1=2, p2=3
    
    Block (i in 0..2, j in m1, k in p1): A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
    This is 3x2 * 2x2 = 3x2, rank = 3*2*2 = 12 (schoolbook) or can we do better?
    
    Actually, let me think differently. The key insight from inspirations:
    Split n=3 as 1+2, m=5 as 2+3, p=5 as 2+3
    
    Then we have blocks:
    - n1=1, m1=2, p1=2: 1x2 * 2x2, rank 4 (schoolbook)
    - n1=1, m1=2, p2=3: 1x2 * 2x3, rank 6
    - n1=1, m2=3, p1=2: 1x3 * 3x2, rank 6
    - n1=1, m2=3, p2=3: 1x3 * 3x3, rank 9
    - n2=2, m1=2, p1=2: 2x2 * 2x2, rank 7 (Strassen!)
    - n2=2, m1=2, p2=3: 2x2 * 2x3, rank 12
    - n2=2, m2=3, p1=2: 2x3 * 3x2, rank 12
    - n2=2, m2=3, p2=3: 2x3 * 3x3, rank 18
    
    Total: 4+6+6+9+7+12+12+18 = 74
    
    That's rank 74, saving 1 from schoolbook's 75!
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    deadline = time.time() + 110.0
    
    # Split indices: n=3 -> n1=1, n2=2; m=5 -> m1=2, m2=3; p=5 -> p1=2, p2=3
    n1, n2 = 1, 2
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Define blocks: (n_block_size, m_block_size, p_block_size, i_start, j_start, k_start)
    blocks = [
        (n1, m1, p1, 0, 0, 0),
        (n1, m1, p2, 0, 0, p1),
        (n1, m2, p1, 0, m1, 0),
        (n1, m2, p2, 0, m1, p1),
        (n2, m1, p1, n1, 0, 0),
        (n2, m1, p2, n1, 0, p1),
        (n2, m2, p1, n1, m1, 0),
        (n2, m2, p2, n1, m1, p1),
    ]
    
    # Initialize full factor matrices
    total_rank = 0
    for (nb, mb, pb, _, _, _) in blocks:
        if nb == 2 and mb == 2 and pb == 2:
            total_rank += 7  # Strassen
        else:
            total_rank += nb * mb * pb  # schoolbook
    
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (nb, mb, pb, i_start, j_start, k_start) in blocks:
        if nb == 2 and mb == 2 and pb == 2:
            U_b, V_b, W_b = _strassen_2x2x2()
            block_rank = 7
        else:
            U_b, V_b, W_b = _schoolbook_block(nb, mb, pb)
            block_rank = nb * mb * pb
        
        # Embed block into full matrices
        for r in range(block_rank):
            # U: A indices are (i, j) where i in [i_start, i_start+nb), j in [j_start, j_start+mb)
            for ii in range(nb):
                for jj in range(mb):
                    U[(i_start+ii)*m + (j_start+jj)][col_offset + r] = U_b[ii*mb+jj][r]
            
            # V: B indices are (j, k) where j in [j_start, j_start+mb), k in [k_start, k_start+pb)
            for jj in range(mb):
                for kk in range(pb):
                    V[(j_start+jj)*p + (k_start+kk)][col_offset + r] = V_b[jj*pb+kk][r]
            
            # W: C indices are (k, i) where k in [k_start, k_start+pb), i in [i_start, i_start+nb)
            # W[k*n + i, r]
            for kk in range(pb):
                for ii in range(nb):
                    W[(k_start+kk)*n + (i_start+ii)][col_offset + r] = W_b[kk*nb+ii][r]
        
        col_offset += block_rank
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        random.seed(42)
        max_err = 0.0
        for _ in range(5):
            A = [[random.gauss(0,1) for _ in range(m)] for _ in range(n)]
            B = [[random.gauss(0,1) for _ in range(p)] for _ in range(m)]
            C_true = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            C_dec = [[0.0]*p for _ in range(n)]
            for c in range(rank):
                a_c = [sum(U[i*m+j][c]*A[i][j] for j in range(m)) for i in range(n)]
                b_c = [sum(V[j*p+k][c]*B[j][k] for j in range(m)) for k in range(p)]
                for i in range(n):
                    for k in range(p):
                        C_dec[i][k] += W[k*n+i][c] * a_c[i] * b_c[k]
            for i in range(n):
                for k in range(p):
                    max_err = max(max_err, abs(C_true[i][k] - C_dec[i][k]))
        return max_err < tol
    
    if verify(U, V, W):
        return {'U': U, 'V': V, 'W': W}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
