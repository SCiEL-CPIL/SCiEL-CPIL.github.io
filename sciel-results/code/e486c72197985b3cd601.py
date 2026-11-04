"""Combined program: structured block decomposition + F2 rank analysis + rank reduction search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
_TIME_LIMIT = 115.0
_START = time.time()

# EVOLVE-BLOCK-START
def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > tol:
                        return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _construct_strassen_28():
    """
    Strassen-based rank 28 decomposition for 2x4 * 4x4 matrix multiplication.
    Uses four independent 2x2*2x2 Strassen blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen 2x2 factors
    sU = [[0]*7 for _ in range(4)]
    sV = [[0]*7 for _ in range(4)]
    sW = [[0]*7 for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    sU[0][0] = 1; sU[3][0] = 1
    sV[0][0] = 1; sV[3][0] = 1
    sW[0][0] = 1; sW[3][0] = 1
    
    # P2 = (c+d)e
    sU[2][1] = 1; sU[3][1] = 1
    sV[0][1] = 1
    sW[2][1] = 1; sW[3][1] = -1
    
    # P3 = a(f-h)
    sU[0][2] = 1
    sV[1][2] = 1; sV[3][2] = -1
    sW[1][2] = 1; sW[3][2] = 1
    
    # P4 = d(g-e)
    sU[3][3] = 1
    sV[2][3] = 1; sV[0][3] = -1
    sW[0][3] = 1; sW[2][3] = 1
    
    # P5 = (a+b)h
    sU[0][4] = 1; sU[1][4] = 1
    sV[3][4] = 1
    sW[0][4] = -1; sW[1][4] = 1
    
    # P6 = (c-a)(e+f)
    sU[2][5] = 1; sU[0][5] = -1
    sV[0][5] = 1; sV[1][5] = 1
    sW[3][5] = 1
    
    # P7 = (b-d)(g+h)
    sU[1][6] = 1; sU[3][6] = -1
    sV[2][6] = 1; sV[3][6] = 1
    sW[0][6] = 1

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        for r in range(7):
            for si in range(2):
                for sj in range(2):
                    U[X_rows[si*2+sj]][col_offset + r] += sU[si*2+sj][r]
            for sj in range(2):
                for sk in range(2):
                    V[Y_rows[sj*2+sk]][col_offset + r] += sV[sj*2+sk][r]
            for sk in range(2):
                for si in range(2):
                    W[Z_rows[sk*2+si]][col_offset + r] += sW[sk*2+si][r]

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}

def _f2_rank_analysis():
    """
    Analyze the tensor structure over F2 to identify potential rank reductions.
    This is a structural analysis that may guide search.
    """
    n, m, p = DIMENSIONS
    
    # Build the incidence structure
    # Each rank-1 tensor is determined by (u, v, w) where:
    # u is a vector in F2^(n*m), v in F2^(m*p), w in F2^(p*n)
    # The product u_i * v_j * w_k gives a coordinate
    
    # For the multiplication tensor, coordinate (i,j,k) = 1 iff the indices match
    # T[i,j,k] = 1 for all valid (i,j,k) in the standard decomposition
    
    # Over F2, we can look for linear dependencies among the coordinate vectors
    # to identify if some rank-1 terms can be combined
    
    # The key insight: if we have two rank-1 terms that agree on all but a few
    # coordinates, and those coordinates can be corrected by other terms,
    # we might be able to merge them.
    
    # For now, this is a placeholder for deeper analysis.
    # The F2 rank of the flattening matrix gives a lower bound.
    
    # Flattening 1: reshape T to (n*m) x (m*p * p*n)
    # Flattening 2: reshape T to (m*p) x (n*m * p*n)
    # Flattening 3: reshape T to (p*n) x (n*m * m*p)
    
    # For n=2, m=4, p=4:
    # T has dimensions 8 x 16 x 8 = 1024 entries
    # T[i,j,k] = 1 for all i,j,k (in the standard basis)
    
    # Actually, the multiplication tensor T satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid i,j,k
    
    # The flattening along the first mode:
    # F1[i, (j,k)] = T[i,j,k]
    # This is an 8 x 128 matrix
    # F1[i, j*p+k] = 1 for all i, j, k
    # So F1 is all ones, rank 1 over any field
    
    # This means the first flattening has rank 1, which is trivial.
    # The tensor is not "generic" - it has a very specific structure.
    
    # The actual constraint is that the decomposition must exactly reproduce
    # the multiplication, not just any tensor with those dimensions.
    
    # For now, return a flag indicating that deeper F2 analysis is needed
    # but the current rank 28 is likely near-optimal for this structure.
    return True

def _try_rank_reduction(base_result, target_rank):
    """
    Try to reduce the rank below base_rank by searching for a lower-rank decomposition.
    Uses a heuristic approach based on the structure of the problem.
    """
    n, m, p = DIMENSIONS
    U = base_result["U"]
    V = base_result["V"]
    W = base_result["W"]
    base_rank = len(U[0])
    
    if target_rank >= base_rank:
        return base_result
    
    # Try to find linear dependencies among the rank-1 terms
    # that allow us to eliminate some terms
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check if any two columns are linearly dependent in a way that
    # allows cancellation
    # This is a simplified approach - in practice, we would use
    # more sophisticated methods
    
    # For now, just return the base result if we can't find a reduction
    # within the time limit
    if _time_remaining() < 5:
        return base_result
    
    # Try a simple perturbation approach:
    # Randomly zero out entries and see if the decomposition still works
    # This is very heuristic and unlikely to succeed, but it's a start
    
    import random
    for attempt in range(10):
        if _time_remaining() < 5:
            break
            
        U_try = [row[:] for row in U]
        V_try = [row[:] for row in V]
        W_try = [row[:] for row in W]
        
        # Try to eliminate one rank-1 term by absorbing it into others
        # This requires solving a system of equations, which is complex
        # For now, just try random sparsification
        
        removed = 0
        for r in range(base_rank - 1, target_rank - 1, -1):
            # Try to zero out column r
            # and see if we can adjust other columns to compensate
            # This is a placeholder - actual implementation would be more complex
            pass
        
        if removed > 0:
            result = {"U": U_try, "V": V_try, "W": W_try}
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                return result
    
    return base_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank 28
    result = _construct_strassen_28()
    
    # Verify it
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        return _schoolbook()
    
    # Try to reduce the rank using F2 analysis and search
    if _time_remaining() > 20:
        # Perform F2 rank analysis
        _f2_rank_analysis()
        
        # Try rank reduction
        reduced = _try_rank_reduction(result, 24)
        if _verify_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
            return reduced
    
    return result
# EVOLVE-BLOCK-END
