"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 decomposition."""
import time
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Row 0: schoolbook (24 terms)
    - Rows 1,2: Strassen on 2x2x2 blocks (7 terms per block, 6 blocks = 42 terms)
    - Total rank: 66
    
    U shape: (12, rank)  -- rows indexed by i*4+j
    V shape: (24, rank)  -- rows indexed by j*6+k
    W shape: (18, rank)  -- rows indexed by k*3+i
    """
    n, m, p = 3, 4, 6
    
    # Total rank: 24 (schoolbook row 0) + 42 (Strassen rows 1,2) = 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        """
        Add Strassen rank-7 decomposition for a 2x2x2 sub-block.
        
        i_base: base row index (for rows i_base, i_base+1)
        j_base: base column index of A / row index of B (for cols j_base, j_base+1)
        k_base: base column index of B / column index of C (for cols k_base, k_base+1)
        
        A sub-block:
          A[i_base, j_base]    A[i_base, j_base+1]
          A[i_base+1, j_base]  A[i_base+1, j_base+1]
        
        B sub-block:
          B[j_base, k_base]    B[j_base, k_base+1]
          B[j_base+1, k_base]  B[j_base+1, k_base+1]
        
        C sub-block:
          C[i_base, k_base]    C[i_base, k_base+1]
          C[i_base+1, k_base]  C[i_base+1, k_base+1]
        
        Strassen products:
        P1 = (A11+A22)(B11+B22)
        P2 = (A21+A22)B11
        P3 = A11(B21-B22)
        P4 = A22(B12-B11)
        P5 = (A11+A12)B22
        P6 = (A21-A11)(B11+B12)
        P7 = (A12-A22)(B21+B22)
        
        C11 = P1 - P4 + P7
        C12 = P3 + P5
        C21 = P2 + P6
        C22 = P1 - P3 + P4
        """
        nonlocal col
        
        # Index mappings
        a11_idx = i_base * m + j_base
        a12_idx = i_base * m + (j_base + 1)
        a21_idx = (i_base + 1) * m + j_base
        a22_idx = (i_base + 1) * m + (j_base + 1)
        
        b11_idx = j_base * p + k_base
        b12_idx = j_base * p + (k_base + 1)
        b21_idx = (j_base + 1) * p + k_base
        b22_idx = (j_base + 1) * p + (k_base + 1)
        
        c11_idx = k_base * n + i_base
        c12_idx = (k_base + 1) * n + i_base
        c21_idx = k_base * n + (i_base + 1)
        c22_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (A11+A22)(B11+B22)
        # C11 += P1, C22 += P1
        U[a11_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # P2 = (A21+A22)B11
        # C21 += P2
        U[a21_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # P3 = A11(B21-B22)
        # C12 += P3, C22 -= P3
        U[a11_idx][col] = 1
        V[b21_idx][col] = 1
        V[b22_idx][col] = -1
        W[c12_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # P4 = A22(B12-B11)
        # C11 -= P4, C22 += P4
        U[a22_idx][col] = 1
        V[b12_idx][col] = 1
        V[b11_idx][col] = -1
        W[c11_idx][col] = -1
        W[c22_idx][col] = 1
        col += 1
        
        # P5 = (A11+A12)B22
        # C12 += P5
        U[a11_idx][col] = 1
        U[a12_idx][col] = 1
        V[b22_idx][col] = 1
        W[c12_idx][col] = 1
        col += 1
        
        # P6 = (A21-A11)(B11+B12)
        # C21 += P6
        U[a21_idx][col] = 1
        U[a11_idx][col] = -1
        V[b11_idx][col] = 1
        V[b12_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # P7 = (A12-A22)(B21+B22)
        # C11 += P7
        U[a12_idx][col] = 1
        U[a22_idx][col] = -1
        V[b21_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1

    # Row 0: schoolbook (24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Rows 1,2: Strassen on 2x2x2 blocks
    # 2 j-blocks (j=0,1 and j=2,3) x 3 k-blocks (k=0,1 and k=2,3 and k=4,5)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            add_strassen(1, j_base, k_base)
    
    return {"U": U, "V": V, "W": W}
