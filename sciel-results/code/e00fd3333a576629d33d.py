"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    rank = len(U_f[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != 1:
                    return False
    return True

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    r = round(2 * x) / 2
    if abs(x - r) < tol:
        return r
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2(n, m, p):
    """
    Return a Strassen-style decomposition for n=2, m=2, p=2 with rank 7.
    This is the standard Strassen matrix multiplication algorithm.
    """
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: (i,j) for i in {0,1}, j in {0,1}
    # V rows: (j,k) for j in {0,1}, k in {0,1}
    # W rows: (k,i) for k in {0,1}, i in {0,1}
    
    # M1: A11 + A22 times B11 + B22
    # U contributions: U[0,0]=1, U[1,1]=1
    # V contributions: V[0,0]=1, V[1,1]=1
    # W contributions: W[0,0]=1, W[1,1]=1, W[0,1]=-1, W[1,0]=-1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = -1
    
    # M2: (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A11(B21 - B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22(B11 - B21)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decs, n, m, p):
    """
    Combine multiple decompositions (each for sub-problems) into one.
    decs: list of dicts with 'U', 'V', 'W' and 'offset' info.
    For simplicity, we handle the case where we split p into blocks.
    """
    # This function is a placeholder for a more general combination.
    # For now, we'll handle specific cases in construct().
    raise NotImplementedError("General combination not implemented yet")

def _strassen_plus_schoolbook(n, m, p):
    """
    For n=2, m=3, p=10:
    Split the 3 columns into 2+1.
    Use Strassen (rank 7) for the 2x2x2 sub-problem (columns 0,1 of B, rows 0,1 of A).
    Use schoolbook (rank 2*1*10 = 20... wait, need to be more careful).
    
    Actually, let's split differently. The output C is 2x10.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's split the A matrix into A_left (2x2, columns 0,1) and A_right (2x1, column 2).
    Let's split B correspondingly: B_left (2x10, rows 0,1) and B_right (1x10, row 2).
    
    C = A_left * B_left + A_right * B_right
    
    The first term is a 2x2 * 2x10 multiplication. We can view this as 5 independent 2x2*2x2 multiplications
    if we split the 10 columns into 5 groups of 2. Each uses rank 7. Total rank = 5*7 = 35.
    
    The second term is a 2x1 * 1x10 multiplication. This is just outer products.
    C2[i,k] = A[i,2] * B[2,k]
    This has rank 2*10 = 20 (each (i,k) pair is independent).
    
    Total rank = 35 + 20 = 55.
    """
    rank_strassen = 7 * 5  # 5 blocks of 2x2x2
    rank_schoolbook = 2 * 1 * 10  # 2x1 * 1x10
    
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: Strassen for A_left * B_left
    # A_left is 2x2 (rows 0,1 of A; cols 0,1 of A)
    # B_left is 2x10 (rows 0,1 of B; all 10 cols)
    # We split B_left into 5 blocks of 2 columns each.
    
    col_offset = 0
    for block in range(5):
        # B_left block: rows 0,1; cols 2*block, 2*block+1
        dec = _strassen_2x2(2, 2, 2)
        
        for c in range(7):
            for i in range(2):
                # A row i, col 0 or 1 (since A_left has cols 0,1)
                U[i * m + 0][col_offset + c] = dec['U'][i * 2 + 0][c]  # A[i,0]
                U[i * m + 1][col_offset + c] = dec['U'][i * 2 + 1][c]  # A[i,1]
            
            for j in range(2):
                for k in range(2):
                    # B row j, col 2*block + k
                    V[j * p + (2 * block + k)][col_offset + c] = dec['V'][j * 2 + k][c]
            
            for k in range(2):
                for i in range(2):
                    # C row i, col 2*block + k
                    W[(2 * block + k) * n + i][col_offset + c] = dec['W'][k * 2 + i][c]
        
        col_offset += 7
    
    # Part 2: Schoolbook for A_right * B_right
    # A_right is 2x1 (rows 0,1 of A; col 2 of A)
    # B_right is 1x10 (row 2 of B; all 10 cols)
    
    for i in range(2):
        for k in range(10):
            # C[i,k] = A[i,2] * B[2,k]
            U[i * m + 2][col_offset] = 1  # A[i,2]
            V[2 * p + k][col_offset] = 1  # B[2,k]
            W[k * n + i][col_offset] = 1  # C[i,k]
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the Strassen+Schoolbook decomposition (rank 55)
    best_result = _strassen_plus_schoolbook(n, m, p)
    best_rank = 55
    
    # Verify it's correct
    if not _verify_exact(best_result['U'], best_result['V'], best_result['W'], n, m, p):
        # Fallback to schoolbook if verification fails
        best_result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try to find a better decomposition using optimization
    # For now, we return the best we have (rank 55)
    
    return best_result
# EVOLVE-BLOCK-END
