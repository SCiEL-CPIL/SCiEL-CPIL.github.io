"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    
    For C = A * B where A is 2x2 and B is 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U factors: shape (4, 7), rows indexed by (i,j) in A
    # V factors: shape (4, 7), rows indexed by (i,j) in B  
    # W factors: shape (4, 7), rows indexed by (i,j) in C (output)
    
    # A indices: 0=(0,0)=a, 1=(0,1)=b, 2=(1,0)=c, 3=(1,1)=d
    # B indices: 0=(0,0)=e, 1=(0,1)=f, 2=(1,0)=g, 3=(1,1)=h
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11 += M1
    W[3, 0] = 1  # C22 += M1
    
    # M2 = (c+d)e: U has c=1,d=1; V has e=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21 += M2
    W[3, 1] = -1 # C22 -= M2
    
    # M3 = a(f-h): U has a=1; V has f=1,h=-1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12 += M3
    W[3, 2] = 1  # C22 += M3
    
    # M4 = d(g-e): U has d=1; V has g=1,e=-1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11 += M4
    W[2, 3] = 1  # C21 += M4
    
    # M5 = (a+b)h: U has a=1,b=1; V has h=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11 -= M5
    W[1, 4] = 1  # C12 += M5
    
    # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    
    The 2x16 matrix B has columns indexed 0..15. We group them into
    8 pairs: (col 2t, col 2t+1) for t=0..7.
    
    For each pair t, we treat B[:, 2t:2t+2] as a 2x2 matrix and apply
    Strassen. The output C is 2x16, where C[:, 2t:2t+2] = A * B[:, 2t:2t+2].
    
    U shape: (n*m, rank) = (4, 56)
    V shape: (m*p, rank) = (32, 56)
    W shape: (p*n, rank) = (32, 56)
    
    Indexing conventions:
    - U row index: i*m + j where (i,j) indexes A (i=0,1; j=0,1)
    - V row index: j*p + k where (j,k) indexes B (j=0,1; k=0..15)
    - W row index: k*n + i where (k,i) indexes C output (k=0..15; i=0,1)
      Per the trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7  # column offset in the rank dimension
        
        for r in range(7):
            # U: same for all pairs, just shifted in column
            U[:, col_offset + r] = U_core[:, r]
            
            # V: B row index is j*p + k. For pair t, columns are 2t and 2t+1.
            # V_core has rows 0,1,2,3 corresponding to (j=0,k=0),(j=0,k=1),(j=1,k=0),(j=1,k=1)
            # In the full V, row j*p + k for k in {2t, 2t+1}
            # V_core row 0 -> j=0, local_k=0 -> full_k=2t -> row 0*p + 2t = 2t
            # V_core row 1 -> j=0, local_k=1 -> full_k=2t+1 -> row 0*p + 2t+1 = 2t+1
            # V_core row 2 -> j=1, local_k=0 -> full_k=2t -> row 1*p + 2t = 16+2t
            # V_core row 3 -> j=1, local_k=1 -> full_k=2t+1 -> row 1*p + 2t+1 = 16+2t+1
            V[2*t, col_offset + r] = V_core[0, r]
            V[2*t + 1, col_offset + r] = V_core[1, r]
            V[16 + 2*t, col_offset + r] = V_core[2, r]
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r]
            
            # W: output C row index is k*n + i. For pair t, output columns are 2t and 2t+1.
            # W_core has rows 0,1,2,3 corresponding to C(0,0),C(0,1),C(1,0),C(1,1)
            # W_core row 0 -> C(0,0): i=0, local_k=0 -> full_k=2t -> row 2t*n + 0 = 2t*2 = 4t
            # W_core row 1 -> C(0,1): i=0, local_k=1 -> full_k=2t+1 -> row (2t+1)*2 + 0 = 4t+2
            # W_core row 2 -> C(1,0): i=1, local_k=0 -> full_k=2t -> row 2t*2 + 1 = 4t+1
            # W_core row 3 -> C(1,1): i=1, local_k=1 -> full_k=2t+1 -> row (2t+1)*2 + 1 = 4t+3
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 2, col_offset + r] = W_core[1, r]
            W[4*t + 1, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
    
    # Convert to lists of lists with integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
