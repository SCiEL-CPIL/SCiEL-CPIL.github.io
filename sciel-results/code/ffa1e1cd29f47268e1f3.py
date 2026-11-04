"""Rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

# Strassen 2x2x2 factors (rank 7)
# M1 = a+d, M2 = c+d, M3 = a, M4 = d, M5 = a+b, M6 = c-a, M7 = b-d
STRASSEN_U = [
    [1, 0, 1, 0, 1, -1, 0],  # a
    [0, 0, 0, 0, 1, 0, 1],   # b
    [0, 1, 0, 0, 0, 1, 0],   # c
    [1, 1, 0, 1, 0, 0, -1],  # d
]
STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],  # e
    [0, 0, 1, 0, 0, 1, 0],   # f
    [0, 0, 0, 1, 0, 0, 1],   # g
    [1, 0, -1, 0, 1, 0, 1],  # h
]
STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],  # C11
    [0, 0, 1, 0, 1, 0, 0],   # C12
    [0, 1, 0, 1, 0, 0, 0],   # C21
    [1, -1, 1, 0, 0, -1, 0], # C22
]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    
    Strategy: Block decomposition where possible 2x2x2 blocks use Strassen (rank 7)
    and remaining elements use schoolbook (rank 1 each).
    
    For 3x4*4x7:
    - Use Strassen for the 2x2 block in rows 0-1, cols 0-1 of C (covering i=0,1; k=0,1)
      with j range 0-1 (m-dim indices 0,1)
    - Schoolbook for all other (i,j,k) triples
    
    The 2x2x2 Strassen block covers:
      i in {0,1}, j in {0,1}, k in {0,1} -> 8 schoolbook terms -> 7 Strassen terms
    
    Total: 84 - 8 + 7 = 83... wait, let me recount.
    
    Actually for 3x4*4x7: n=3, m=4, p=7.
    Schoolbook rank = 3*4*7 = 84.
    
    We can apply Strassen to any 2x2x2 sub-block where we have 2 rows of A (i-indices),
    2 columns of A / 2 rows of B (j-indices), and 2 columns of B / 2 rows of C (k-indices).
    
    Let's use Strassen for the block i in {0,1}, j in {0,1}, k in {0,1}.
    That replaces 8 rank-1 terms with 7 rank-1 terms.
    
    We can also use Strassen for other 2x2x2 blocks:
    - i in {0,1}, j in {2,3}, k in {0,1}
    - i in {0,1}, j in {0,1}, k in {2,3}
    - i in {0,1}, j in {2,3}, k in {2,3}
    - i in {0,1}, j in {0,1}, k in {4,5}
    - i in {0,1}, j in {2,3}, k in {4,5}
    - i in {0,1}, j in {0,1}, k in {6} - only 1 k, can't use Strassen
    - etc.
    
    Let me think about this more carefully. For each pair of i-indices (2 consecutive),
    pair of j-indices (2 consecutive), and pair of k-indices (2 consecutive), we can
    apply Strassen.
    
    i-pairs: (0,1), (1,2) - 2 pairs
    j-pairs: (0,1), (2,3) - 2 pairs  
    k-pairs: (0,1), (2,3), (4,5) - 3 pairs
    
    But these blocks must be disjoint (no overlapping indices) to simply sum ranks.
    
    Disjoint tiling approach:
    - i: use (0,1) for Strassen, i=2 is remainder
    - j: use (0,1) and (2,3) for Strassen
    - k: use (0,1), (2,3), (4,5) for Strassen, k=6 is remainder
    
    Strassen blocks:
    (i:0-1, j:0-1, k:0-1) -> 7
    (i:0-1, j:2-3, k:0-1) -> 7
    (i:0-1, j:0-1, k:2-3) -> 7
    (i:0-1, j:2-3, k:2-3) -> 7
    (i:0-1, j:0-1, k:4-5) -> 7
    (i:0-1, j:2-3, k:4-5) -> 7
    Total Strassen: 6*7 = 42
    
    Remainder schoolbook:
    i=2 with all j,k: 4*7 = 28
    i=0,1 with k=6: 2*4*1 = 8
    Total remainder: 28 + 8 = 36
    
    Total rank: 42 + 36 = 78
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_rows = n * m  # 12
    V_rows = m * p  # 28
    W_rows = p * n  # 21
    
    # We'll build the decomposition by accumulating rank-1 terms
    # Each term is (U_row, V_row, W_row, coefficient)
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    # Define Strassen block decomposition
    def add_strassen_block(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)."""
        for r in range(4):  # r: 0=a, 1=b, 2=c, 3=d for U; 0=e,1=f,2=g,3=h for V; 0=C11,1=C12,2=C21,3=C22 for W
            for c in range(7):
                coeff_u = STRASSEN_U[r][c]
                coeff_v = STRASSEN_V[r][c]
                coeff_w = STRASSEN_W[r][c]
                if coeff_u != 0 and coeff_v != 0 and coeff_w != 0:
                    # U row: i = i0 + (0 if r in [0,1] else 1), j = j0 + (0 if r in [0,2] else 1)
                    # Wait, the Strassen decomposition maps:
                    # a = A[i0, j0], b = A[i0, j0+1], c = A[i0+1, j0], d = A[i0+1, j0+1]
                    # So U row index: for M1..M7, the U coefficient is for a specific (i,j)
                    # 
                    # Actually, the standard Strassen factorization:
                    # M1 = (a+d)(e+h): U has a:1, d:1; V has e:1, h:1; W C11:1, C22:1
                    # This means each M_k is a rank-1 term in the tensor.
                    # 
                    # The U matrix row corresponds to (i,j) index, V to (j,k), W to (k,i)
                    # 
                    # For the block starting at (i0, j0, k0):
                    # a = A[i0, j0]   -> U row = i0*m + j0
                    # b = A[i0, j0+1] -> U row = i0*m + (j0+1)
                    # c = A[i0+1, j0] -> U row = (i0+1)*m + j0
                    # d = A[i0+1, j0+1] -> U row = (i0+1)*m + (j0+1)
                    # 
                    # e = B[j0, k0]   -> V row = j0*p + k0
                    # f = B[j0, k0+1] -> V row = j0*p + (k0+1)
                    # g = B[j0+1, k0] -> V row = (j0+1)*p + k0
                    # h = B[j0+1, k0+1] -> V row = (j0+1)*p + (k0+1)
                    # 
                    # C[i0, k0]   -> W row = k0*n + i0
                    # C[i0, k0+1] -> W row = (k0+1)*n + i0
                    # C[i0+1, k0] -> W row = k0*n + (i0+1)
                    # C[i0+1, k0+1] -> W row = (k0+1)*n + (i0+1)
                    # 
                    # The 7 Strassen multiplications:
                    # M1 = (a+d)(e+h)
                    # M2 = (c+d)(e)
                    # M3 = (a)(f-h)
                    # M4 = (d)(g-e)
                    # M5 = (a+b)(h)
                    # M6 = (c-a)(e+f)
                    # M7 = (b-d)(g+h)
                    # 
                    # Output:
                    # C[i0,k0] = M1 + M4 - M5 + M7
                    # C[i0,k0+1] = M3 + M5
                    # C[i0+1,k0] = M2 + M4
                    # C[i0+1,k0+1] = M1 - M2 + M3 - M6
                    # 
                    # Each M_k is a rank-1 term. For each M_k, we need to add:
                    # U contribution: sum over (i,j) of coeff * e_{i,j}
                    # V contribution: sum over (j,k) of coeff * e_{j,k}
                    # W contribution: sum over (k,i) of coeff * e_{k,i}
                    # 
                    # But since we're building the full U, V, W matrices, each rank-1 term
                    # contributes to specific rows.
                    pass
        
        # Actually, let me just build the full U, V, W directly.
        # For each Strassen multiplication M_k, it's a rank-1 tensor:
        # T = u_vec * v_vec * w_vec where u_vec, v_vec, w_vec are vectors
        # 
        # The U matrix will have columns corresponding to each rank-1 term.
        # U[row, col] = coefficient of that row in the U vector of the col-th term.
        
        # Let me restructure: build lists of (u_row, coeff), (v_row, coeff), (w_row, coeff) for each term
        pass
    
    # Simpler approach: directly construct U, V, W as lists of lists
    # First, determine total rank
    # Strassen blocks: (i:0-1, j:0-1, k:0-1), (i:0-1, j:2-3, k:0-1), 
    # (i:0-1, j:0-1, k:2-3), (i:0-1, j:2-3, k:2-3), (i:0-1, j:0-1, k:4-5), (i:0-1, j:2-3, k:4-5)
    # Each has rank 7, total 42
    # Remainder: i=2 (all j,k) = 28, i=0,1 with k=6 (all j) = 8, total 36
    # Total rank = 78
    
    rank = 78
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Add Strassen blocks
    strassen_blocks = [
        (0, 0, 0),  # i:0-1, j:0-1, k:0-1
        (0, 2, 0),  # i:0-1, j:2-3, k:0-1
        (0, 0, 2),  # i:0-1, j:0-1, k:2-3
        (0, 2, 2),  # i:0-1, j:2-3, k:2-3
        (0, 0, 4),  # i:0-1, j:0-1, k:4-5
        (0, 2, 4),  # i:0-1, j:2-3, k:4-5
    ]
    
    for (i0, j0, k0) in strassen_blocks:
        # For each of the 7 Strassen multiplications
        # M1 = (a+d)(e+h)
        # U: a at (i0,j0) coeff 1, d at (i0+1,j0+1) coeff 1
        # V: e at (j0,k0) coeff 1, h at (j0+1,k0+1) coeff 1
        # W: C[i0,k0] coeff 1, C[i0+1,k0+1] coeff 1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2 = (c+d)(e)
        # U: c at (i0+1,j0) coeff 1, d at (i0+1,j0+1) coeff 1
        # V: e at (j0,k0) coeff 1
        # W: C[i0+1,k0] coeff 1
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M3 = (a)(f-h)
        # U: a at (i0,j0) coeff 1
        # V: f at (j0,k0+1) coeff 1, h at (j0+1,k0+1) coeff -1
        # W: C[i0,k0+1] coeff 1
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4 = (d)(g-e)
        # U: d at (i0+1,j0+1) coeff 1
        # V: g at (j0+1,k0) coeff 1, e at (j0,k0) coeff -1
        # W: C[i0,k0] coeff 1, C[i0+1,k0] coeff 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (a+b)(h)
        # U: a at (i0,j0) coeff 1, b at (i0,j0+1) coeff 1
        # V: h at (j0+1,k0+1) coeff 1
        # W: C[i0,k0] coeff -1, C[i0,k0+1] coeff 1
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # U: c at (i0+1,j0) coeff 1, a at (i0,j0) coeff -1
        # V: e at (j0,k0) coeff 1, f at (j0,k0+1) coeff 1
        # W: C[i0+1,k0+1] coeff -1
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        
        # M7 = (b-d)(g+h)
        # U: b at (i0,j0+1) coeff 1, d at (i0+1,j0+1) coeff -1
        # V: g at (j0+1,k0) coeff 1, h at (j0+1,k0+1) coeff 1
        # W: C[i0,k0] coeff 1
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
    
    # Add schoolbook terms for remainder
    # i=2, all j in 0..3, all k in 0..6
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # i=0,1 with k=6, all j in 0..3
    for i in range(2):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
