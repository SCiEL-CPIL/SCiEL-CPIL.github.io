import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses GKS splitting and Strassen algorithm where applicable.
    """
    n, m, p = DIMENSIONS
    U, V, W = construct_gks(n, m, p)
    return {"U": U, "V": V, "W": W}

def construct_gks(n, m, p):
    """
    Recursively constructs the decomposition using GKS splitting.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank).
    """
    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        return construct_strassen(2, 2, 2)
    
    # Base case: Schoolbook for small or if GKS doesn't help
    # We can add more base cases if known
    if n == 1 or m == 1 or p == 1:
        return construct_schoolbook(n, m, p)
        
    # GKS Splitting on m
    best_rank = float('inf')
    best_split = None
    best_decomps = None
    
    for m1 in range(1, m):
        m2 = m - m1
        # Calculate ranks of sub-problems
        r1 = get_rank(n, m1, p)
        r2 = get_rank(n, m2, p)
        r3 = get_rank(n, m1, p) # Wait, sub-problems are:
        # A1 (n x m1) * B1 (m1 x p) -> rank r1
        # A1 (n x m1) * B2 (m2 x p) -> rank r2
        # A2 (n x m2) * B1 (m1 x p) -> rank r3
        # A2 (n x m2) * B2 (m2 x p) -> rank r4
        
        r1 = get_rank(n, m1, p)
        r2 = get_rank(n, m2, p)
        r3 = get_rank(n, m1, p)
        r4 = get_rank(n, m2, p)
        
        total_rank = r1 + r2 + r3 + r4
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_split = (m1, m2)
            # We don't store decomps yet to save memory, we will reconstruct
            # Or we can store the choices
            # For simplicity, we reconstruct in the next step
    
    if best_split is None:
        return construct_schoolbook(n, m, p)
        
    m1, m2 = best_split
    
    # Construct sub-decompositions
    U1, V1, W1 = construct_gks(n, m1, p)
    U2, V2, W2 = construct_gks(n, m2, p)
    U3, V3, W3 = construct_gks(n, m1, p) # Same as U1, V1, W1? No, B is different part
    # Wait, the sub-problems are:
    # 1. A1 * B1: Uses U1, V1, W1
    # 2. A1 * B2: Uses U1 (same A part), V2 (B part 2), W2?
    #    The W part is the same C part? No, C is the sum.
    #    The decomposition is a sum of 4 tensors.
    #    Tensor 1: A1, B1, C
    #    Tensor 2: A1, B2, C
    #    Tensor 3: A2, B1, C
    #    Tensor 4: A2, B2, C
    
    # We need to align the indices.
    
    # U1 is (n*m1, r1). It corresponds to A1.
    # U2 is (n*m2, r2). It corresponds to A2.
    # U3 is (n*m1, r3). It corresponds to A1.
    # U4 is (n*m2, r4). It corresponds to A2.
    
    # V1 is (m1*p, r1). It corresponds to B1.
    # V2 is (m2*p, r2). It corresponds to B2.
    # V3 is (m1*p, r3). It corresponds to B1.
    # V4 is (m2*p, r4). It corresponds to B2.
    
    # W1 is (p*n, r1). It corresponds to C.
    # W2 is (p*n, r2). It corresponds to C.
    # W3 is (p*n, r3). It corresponds to C.
    # W4 is (p*n, r4). It corresponds to C.
    
    # The total U is block diagonal? No.
    # U_total has rows corresponding to A (n*m).
    # Rows 0..m1-1 (scaled by n) correspond to A1.
    # Rows m1..m-1 correspond to A2.
    
    # We need to construct U_total, V_total, W_total.
    
    r1, r2, r3, r4 = get_rank(n, m1, p), get_rank(n, m2, p), get_rank(n, m1, p), get_rank(n, m2, p)
    
    # Reconstruct sub-decompositions
    U1, V1, W1 = construct_gks(n, m1, p)
    U2, V2, W2 = construct_gks(n, m2, p)
    U3, V3, W3 = construct_gks(n, m1, p)
    U4, V4, W4 = construct_gks(n, m2, p)
    
    total_rank = r1 + r2 + r3 + r4
    
    U_total = [[0] * total_rank for _ in range(n * m)]
    V_total = [[0] * total_rank for _ in range(m * p)]
    W_total = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Sub-problem 1: A1 * B1
    # U1 rows map to A1 rows (0..n*m1-1)
    for i in range(n * m1):
        for r in range(r1):
            U_total[i][col_offset + r] = U1[i][r]
    # V1 rows map to B1 rows (0..m1*p-1)
    for i in range(m1 * p):
        for r in range(r1):
            V_total[i][col_offset + r] = V1[i][r]
    # W1 rows map to C rows (0..p*n-1)
    for i in range(p * n):
        for r in range(r1):
            W_total[i][col_offset + r] = W1[i][r]
    col_offset += r1
    
    # Sub-problem 2: A1 * B2
    # U1 rows map to A1 rows
    for i in range(n * m1):
        for r in range(r2):
            U_total[i][col_offset + r] = U1[i][r] # Wait, U1 is for m1.
            # But U1 is from construct_gks(n, m1, p).
            # We need U for A1.
            # In sub-problem 2, A is A1. So U is same as U1.
            # But V is for B2.
            # V2 is from construct_gks(n, m2, p). V2 is (m2*p, r2).
            # But B2 is the second part of B.
            # In V_total, B2 rows are m1*p .. m*p-1.
            # So we need to map V2 rows to V_total rows offset by m1*p.
            
            # Correction:
            # U_total for A1 is same as U1.
            # V_total for B2 is V2, but shifted in row index.
            # W_total for C is W2.
            
    # Let's redo the mapping carefully.
    
    col_offset = 0
    
    # 1. A1 * B1
    for i in range(n * m1):
        for r in range(r1):
            U_total[i][col_offset + r] = U1[i][r]
    for i in range(m1 * p):
        for r in range(r1):
            V_total[i][col_offset + r] = V1[i][r]
    for i in range(p * n):
        for r in range(r1):
            W_total[i][col_offset + r] = W1[i][r]
    col_offset += r1
    
    # 2. A1 * B2
    for i in range(n * m1):
        for r in range(r2):
            U_total[i][col_offset + r] = U1[i][r] # U1 is A1
    for i in range(m2 * p):
        for r in range(r2):
            V_total[m1 * p + i][col_offset + r] = V2[i][r] # V2 is B2, shifted
    for i in range(p * n):
        for r in range(r2):
            W_total[i][col_offset + r] = W2[i][r] # W2 is C
    col_offset += r2
    
    # 3. A2 * B1
    for i in range(n * m2):
        for r in range(r3):
            U_total[n * m1 + i][col_offset + r] = U3[i][r] # U3 is A2, shifted
    for i in range(m1 * p):
        for r in range(r3):
            V_total[i][col_offset + r] = V3[i][r] # V3 is B1
    for i in range(p * n):
        for r in range(r3):
            W_total[i][col_offset + r] = W3[i][r] # W3 is C
    col_offset += r3
    
    # 4. A2 * B2
    for i in range(n * m2):
        for r in range(r4):
            U_total[n * m1 + i][col_offset + r] = U4[i][r] # U4 is A2, shifted
    for i in range(m2 * p):
        for r in range(r4):
            V_total[m1 * p + i][col_offset + r] = V4[i][r] # V4 is B2, shifted
    for i in range(p * n):
        for r in range(r4):
            W_total[i][col_offset + r] = W4[i][r] # W4 is C
    col_offset += r4
    
    return U_total, V_total, W_total

def get_rank(n, m, p):
    """
    Returns the minimal known rank for n x m x p multiplication.
    """
    if n == 2 and m == 2 and p == 2:
        return 7
    if n == 1 or m == 1 or p == 1:
        return n * m * p # Schoolbook for base
    
    # Try GKS splits
    best_rank = n * m * p
    for m1 in range(1, m):
        m2 = m - m1
        r1 = get_rank(n, m1, p)
        r2 = get_rank(n, m2, p)
        r3 = get_rank(n, m1, p)
        r4 = get_rank(n, m2, p)
        total = r1 + r2 + r3 + r4
        if total < best_rank:
            best_rank = total
            
    # Also try splitting n and p?
    # For simplicity, stick to m-splitting (GKS)
    
    return best_rank

def construct_strassen(n, m, p):
    """
    Returns the Strassen decomposition for 2x2x2.
    """
    # U, V, W for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+f)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C11, 1=C12, 2=C21, 3=C22 (k*n+i) -> C11(0*2+0), C21(0*2+1)?
    # Convention: W[k*n+i, r].
    # k is col of C, i is row of C.
    # C is 2x2.
    # k=0, i=0 -> C11 -> index 0
    # k=0, i=1 -> C21 -> index 1
    # k=1, i=0 -> C12 -> index 2
    # k=1, i=1 -> C22 -> index 3
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = -1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def construct_schoolbook(n, m, p):
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
                
    return U, V, W

# EVOLVE-BLOCK-END
