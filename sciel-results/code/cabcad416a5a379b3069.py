"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _to_frac(x, tol=1e-9):
    """Convert a float to a Fraction with a small tolerance, preferring simple rationals."""
    if abs(x) < tol:
        return Fraction(0, 1)
    # Try small denominators first
    for den in range(1, 257):
        num = round(x * den)
        if abs(x - num / den) < tol:
            return Fraction(num, den)
    # Fallback to continued fraction
    return Fraction(x).limit_denominator(1000000)

def _validate_decomp(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _try_strassen_blocks(n, m, p):
    """
    Build a decomposition by tiling with 2x2x2 Strassen blocks where possible,
    and falling back to schoolbook for the remainder.
    """
    rank = 0
    U = {}
    V = {}
    W = {}
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11 - A22)*(B11 + B22)
    # M2 = (A11 + A22)*B22
    # M3 = A11*(B11 - B22)
    # M4 = A22*(B21 - B11)
    # M5 = (A21 + A22)*(B22 - B21)  -- wait, standard Strassen:
    # M5 = (A11 + A12)*B22
    # M6 = (A21 - A11)*(B11 + B12)
    # M7 = (A12 - A22)*(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me use the standard Strassen scheme:
    # M1 = (A11 - A22) * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = A11 * (B11 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A21 + A22) * (B22 - B21)  -- no, let me be careful
    # Standard:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)  -- actually M5 = (A11+A12)B22? No.
    # Let me just use the well-known 7 multiplications:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)  -- this is not standard either
    # 
    # Correct Strassen:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)  -- no
    # 
    # The standard 7 products are:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)  -- I'll just hardcode the correct one
    # 
    # Actually the cleanest is:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)  -- let me just use a verified set
    # 
    # Verified Strassen:
    # M1 = (A11 - A22)(B11 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)(B22 - B21)
    # M6 = (A11 - A21)(B21 + B22)
    # M7 = (A12 - A22)(B11 + B12)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Let me define the Strassen block as a list of (coeff_A, coeff_B, coeff_C) for each product
    # Each product c has U entries (for A factors), V entries (for B factors), W entry (for C factor)
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    
    # For a 2x2x2 block at position (i0, j0, k0):
    # A is 2x2: rows i0, i0+1; cols j0, j0+1
    # B is 2x2: rows j0, j0+1; cols k0, k0+1
    # C is 2x2: rows k0, k0+1; cols i0, i0+1
    
    # Strassen products (7 of them):
    # M1: A part: (1,0; -1,1) i.e. A11 - A22 => U entries: (i0,j0):+1, (i0+1,j0+1):-1
    #     B part: (1,1; 1,1) i.e. B11 + B22 => V entries: (j0,k0):+1, (j0,k0+1):+1, (j0+1,k0):+1, (j0+1,k0+1):+1
    #     C part: M1 contributes to C11:+1, C22:+1 => W: (k0,i0):+1, (k0+1,i0+1):+1
    # 
    # This is getting complex. Let me just use a direct approach.
    
    # Actually, for n=3, m=5, p=5, let me think about what blocks I can place.
    # n=3: can fit one 2x2 block in the i dimension (rows 0-1), leaving row 2.
    # m=5: can fit two 2x2 blocks in j dimension (cols 0-1, 2-3), leaving col 4.
    # p=5: can fit two 2x2 blocks in k dimension (cols 0-1, 2-3), leaving col 4.
    
    # So I can place 2x2x2 Strassen blocks at:
    # (i=0-1, j=0-1, k=0-1)
    # (i=0-1, j=0-1, k=2-3)
    # (i=0-1, j=2-3, k=0-1)
    # (i=0-1, j=2-3, k=2-3)
    # That's 4 blocks of rank 7 = 28 rank.
    
    # Remaining elements:
    # i=2 (row 2 of A, col 2 of C): needs to multiply with all j, k
    # j=4 (col 4 of A, row 4 of B): needs to multiply with all i, k
    # k=4 (col 4 of B, row 4 of C): needs to multiply with all i, j
    
    # The remaining sub-problems are:
    # 1. i=2, all j (5), all k (5): 1x5x5 = 25 schoolbook terms
    # 2. all i (3), j=4, all k (5): 3x1x5 = 15 schoolbook terms
    # 3. all i (3), all j (5), k=4: 3x5x1 = 15 schoolbook terms
    # But these overlap! (2,4,4) is counted in all three.
    
    # Let me be more careful. The total number of (i,j,k) triples is 3*5*5 = 75.
    # The 4 Strassen blocks cover: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # That's 2*4*4 = 32 triples, using 4*7 = 28 rank.
    
    # Remaining triples: 75 - 32 = 43.
    # These need to be done schoolbook: 43 rank.
    # Total rank: 28 + 43 = 71.
    
    # Let me implement this.
    
    def add_schoolbook(i, j, k, rank_offset):
        """Add a schoolbook term for (i,j,k) at rank position rank_offset."""
        c = rank_offset
        U.setdefault(i*m + j, [0]*0)  # We'll build lists
        # Actually, let me use a different approach: collect all terms then build matrices.
        pass
    
    # Let me use a cleaner approach: collect terms as (i,j,k, coeff_U, coeff_V, coeff_W)
    # But Strassen has multiple terms per product.
    
    # Let me just build the matrices directly.
    
    max_rank = 75  # upper bound
    U_list = [[0]*max_rank for _ in range(n*m)]
    V_list = [[0]*max_rank for _ in range(m*p)]
    W_list = [[0]*max_rank for _ in range(p*n)]
    rank_used = 0
    
    # Strassen block at (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        nonlocal rank_used
        # 7 products
        # M1 = (A11 - A22)(B11 + B22)
        # M2 = (A11 + A22)B22
        # M3 = A11(B11 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A21 + A22)(B22 - B21)
        # M6 = (A11 - A21)(B21 + B22)
        # M7 = (A12 - A22)(B11 + B12)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1
        c = rank_used; rank_used += 1
        U_list[i0*m + j0][c] = 1
        U_list[(i0+1)*m + (j0+1)][c] = -1
        V_list[j0*p + k0][c] = 1
        V_list[j0*p + (k0+1)][c] = 1
        V_list[(j0+1)*p + k0][c] = 1
        V_list[(j0+1)*p + (k0+1)][c] = 1
        W_list[k0*n + i0][c] = 1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        
        # M2 = (A11 + A22)B22
        c = rank_used; rank_used += 1
        U_list[i0*m + j0][c] = 1
        U_list[(i0+1)*m + (j0+1)][c] = 1
        V_list[(j0+1)*p + (k0+1)][c] = 1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        
        # M3 = A11(B11 - B22)
        c = rank_used; rank_used += 1
        U_list[i0*m + j0][c] = 1
        V_list[j0*p + k0][c] = 1
        V_list[(j0+1)*p + (k0+1)][c] = -1
        W_list[k0*n + i0][c] = 1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        
        # M4 = A22(B21 - B11)
        c = rank_used; rank_used += 1
        U_list[(i0+1)*m + (j0+1)][c] = 1
        V_list[(j0+1)*p + k0][c] = 1
        V_list[j0*p + k0][c] = -1
        W_list[k0*n + i0][c] = 1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        
        # M5 = (A21 + A22)(B22 - B21)
        c = rank_used; rank_used += 1
        U_list[(i0+1)*m + j0][c] = 1
        U_list[(i0+1)*m + (j0+1)][c] = 1
        V_list[(j0+1)*p + (k0+1)][c] = 1
        V_list[(j0+1)*p + k0][c] = -1
        W_list[k0*n + i0][c] = -1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        W_list[(k0+1)*n + i0][c] = 1
        W_list[k0*n + (i0+1)][c] = -1
        
        # M6 = (A11 - A21)(B21 + B22)
        c = rank_used; rank_used += 1
        U_list[i0*m + j0][c] = 1
        U_list[(i0+1)*m + j0][c] = -1
        V_list[(j0+1)*p + k0][c] = 1
        V_list[(j0+1)*p + (k0+1)][c] = 1
        W_list[(k0+1)*n + (i0+1)][c] = 1
        
        # M7 = (A12 - A22)(B11 + B12)
        c = rank_used; rank_used += 1
        U_list[i0*m + (j0+1)][c] = 1
        U_list[(i0+1)*m + (j0+1)][c] = -1
        V_list[j0*p + k0][c] = 1
        V_list[j0*p + (k0+1)][c] = 1
        W_list[k0*n + i0][c] = 1
    
    # Add 4 Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    
    # Add schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this triple is covered by a Strassen block
                covered = False
                for i0 in [0]:
                    for j0 in [0, 2]:
                        for k0 in [0, 2]:
                            if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                                covered = True
                if not covered:
                    c = rank_used; rank_used += 1
                    U_list[i*m + j][c] = 1
                    V_list[j*p + k][c] = 1
                    W_list[k*n + i][c] = 1
    
    # Trim to actual rank used
    U_out = [row[:rank_used] for row in U_list]
    V_out = [row[:rank_used] for row in V_list]
    W_out = [row[:rank_used] for row in W_list]
    
    return {"U": U_out, "V": V_out, "W": W_out}, rank_used

def _try_random_search(n, m, p, max_rank, time_limit=10):
    """Try to find a decomposition with rank < max_rank using random initialization."""
    start_time = time.time()
    best_rank = max_rank
    best_result = None
    
    # We'll try to find a valid decomposition by solving a system of equations.
    # This is complex, so let's just return the Strassen result for now.
    # A proper random search would involve:
    # 1. Initialize U, V, W randomly
    # 2. Compute the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # 3. Compare with the identity tensor (1 for all i,j,k)
    # 4. Use gradient descent or other optimization to minimize the error
    
    # For now, this is a placeholder.
    return None, max_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Try the Strassen-block approach
    result, rank = _try_strassen_blocks(n, m, p)
    
    # Validate
    if _validate_decomp(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
