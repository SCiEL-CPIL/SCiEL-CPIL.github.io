"""Matrix multiplication tensor decomposition for 3x3 times 3x5 matrices."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The decomposition is of the form:
    C[i, k] = sum_j sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] * A[i,j] * B[j,k]
    
    Actually, the tensor decomposition is:
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where T[i,j,k] = 1 for all valid i, j, k (since C[i,k] = sum_j A[i,j]B[j,k]).
    
    We use the slice method: fix U to a structured basis, then solve linear
    equations for V and W. Fall back to schoolbook if no solution found.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try the slice method with various rank values
    for rank in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        result = _slice_method(n, m, p, rank)
        if result is not None:
            return result
    
    # Fall back to schoolbook decomposition
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # W[k*n + i, r] convention
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_strassen_u(n, m, p, rank):
    """
    Build a structured U matrix using Strassen's 7-term basis for the 3x3x3 core
    and zero-padding for extra columns.
    
    For the 3x3x3 core (i in 0..2, j in 0..2, k in 0..2), use Strassen's 7 terms.
    For k in 3..4 (extra columns), use additional terms.
    """
    # Strassen's 7-term decomposition for 3x3x3
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U for Strassen terms (i*m + j, r) where r=0..6
    strassen_U = [
        # r=0: (A11+A22) -> U[0*3+0]=1, U[1*3+1]=1
        {0: 1, 4: 1},
        # r=1: (A21+A22) -> U[1*3+1]=1, U[2*3+2]=1
        {4: 1, 8: 1},
        # r=2: A11 -> U[0*3+0]=1
        {0: 1},
        # r=3: A22 -> U[1*3+1]=1
        {4: 1},
        # r=4: (A11+A12) -> U[0*3+0]=1, U[0*3+1]=1
        {0: 1, 1: 1},
        # r=5: (A21-A11) -> U[1*3+1]=-1, U[0*3+0]=1
        {4: -1, 0: 1},
        # r=6: (A12-A22) -> U[0*3+1]=1, U[1*3+1]=-1
        {1: 1, 4: -1},
    ]
    
    # V for Strassen terms (j*p + k, r) where r=0..6
    # For the 3x3x3 core, k only goes 0..2
    strassen_V = [
        # r=0: (B11+B22) -> V[0*3+0]=1, V[1*3+1]=1
        {0: 1, 4: 1},
        # r=1: B11 -> V[0*3+0]=1
        {0: 1},
        # r=2: (B21-B22) -> V[1*3+0]=-1, V[1*3+1]=1
        {3: -1, 4: 1},
        # r=3: (B12-B11) -> V[0*3+1]=1, V[0*3+0]=-1
        {1: 1, 0: -1},
        # r=4: B22 -> V[1*3+1]=1
        {4: 1},
        # r=5: (B11+B12) -> V[0*3+0]=1, V[0*3+1]=1
        {0: 1, 1: 1},
        # r=6: (B21+B22) -> V[1*3+0]=1, V[1*3+1]=1
        {3: 1, 4: 1},
    ]
    
    # W for Strassen terms (k*n + i, r) where r=0..6
    strassen_W = [
        # r=0: C11 += M1, C22 += M1 -> W[0*3+0]=1, W[1*3+1]=1
        {0: 1, 4: 1},
        # r=1: C21 += M2 -> W[0*3+1]=1
        {3: 1},
        # r=2: C12 += M3 -> W[1*3+0]=1
        {1: 1},
        # r=3: C21 += M4 -> W[0*3+1]=1
        {3: 1},
        # r=4: C12 += M5 -> W[1*3+0]=1
        {1: 1},
        # r=5: C11 += M6 -> W[0*3+0]=1
        {0: 1},
        # r=6: C22 += M7 -> W[1*3+1]=1
        {4: 1},
    ]
    
    # Build U matrix of shape (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    
    # Fill in Strassen terms for r=0..6
    for r in range(7):
        for idx, val in strassen_U[r].items():
            U[idx][r] = val
    
    # For extra columns (k=3,4), we need additional terms
    # These will be determined by the linear system
    # For now, leave them as 0 (to be solved)
    
    return U, strassen_V, strassen_W


def _slice_method(n, m, p, rank):
    """
    Use the slice method: fix U to a structured basis, then solve linear
    equations for V and W.
    
    The tensor equation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Given U, this is linear in V and W if we fix one of them.
    We'll use an iterative approach: initialize V, solve for W, then solve for V.
    """
    # Build structured U
    U, V_init, W_init = _build_strassen_u(n, m, p, rank)
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    
    # Initialize V and W with Strassen values for first 7 terms
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for r in range(7):
        for idx, val in V_init[r].items():
            V[idx][r] = Fraction(val)
        for idx, val in W_init[r].items():
            W[idx][r] = Fraction(val)
    
    # For terms r=7..rank-1, we need to determine V and W
    # The equations are: for each (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is still nonlinear in V and W. We need a different approach.
    # Let's use random initialization and iterative refinement.
    
    # Actually, let's try a simpler approach: use the schoolbook structure
    # but with fewer terms by exploiting sparsity.
    
    # For rank < 45, we need to find a valid decomposition.
    # Let's try rank 17 with a specific structure.
    
    if rank < 17:
        return None
    
    # Try to construct a rank-17 decomposition
    if rank == 17:
        return _construct_rank_17(n, m, p)
    
    # For other ranks, fall back to None
    return None


def _construct_rank_17(n, m, p):
    """
    Construct a rank-17 decomposition for 3x3 * 3x5 matrix multiplication.
    
    Structure: 7 Strassen terms for the 3x3x3 core + 10 terms for the extra 2 columns.
    """
    # Strassen's 7-term decomposition for 3x3x3
    # We'll use the standard Strassen basis
    
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms (r=0..6)
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[4][0] = 1  # A22
    V[0][0] = 1  # B11
    V[4][0] = 1  # B22
    W[0][0] = 1  # C11
    W[4][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[4][1] = 1  # A21
    U[8][1] = 1  # A22
    V[0][1] = 1  # B11
    W[3][1] = 1  # C21
    
    # M3 = A11(B21-B22)
    U[0][2] = 1  # A11
    V[3][2] = 1  # B21
    V[4][2] = -1  # B22
    W[1][2] = 1  # C12
    
    # M4 = A22(B12-B11)
    U[4][3] = 1  # A22
    V[1][3] = 1  # B12
    V[0][3] = -1  # B11
    W[3][3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[4][4] = 1  # B22
    W[1][4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[4][5] = 1  # A21
    U[0][5] = -1  # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[0][5] = 1  # C11
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[4][6] = -1  # A22
    V[3][6] = 1  # B21
    V[4][6] = 1  # B22
    W[4][6] = 1  # C22
    
    # Extra columns k=3,4 (r=7..16)
    # For k=3: C[i,3] = sum_j A[i,j] * B[j,3]
    # We need 5 terms for the 3x5 slice (i in 0..2, j in 0..2, k=3)
    # But we can reuse some structure.
    
    # Actually, for the extra columns, we can use a simple linear decomposition.
    # C[i,3] = sum_j A[i,j] * B[j,3]
    # This requires 3*3 = 9 terms in general, but we can do better.
    
    # Let's use 5 terms for k=3 and 5 terms for k=4 (total 10 extra terms)
    # But that gives 7+10=17 total.
    
    # For k=3, use terms r=7..11
    # C[i,3] = sum_j A[i,j] * B[j,3]
    # Term r=7: A[0,0]*B[0,3]
    U[0][7] = 1
    V[0*5+3][7] = 1  # j=0, k=3 -> index 3
    W[3*n+0][7] = 1  # k=3, i=0 -> index 3*3+0=9
    
    # Term r=8: A[0,1]*B[1,3]
    U[1][8] = 1
    V[1*5+3][8] = 1  # j=1, k=3 -> index 8
    W[3*n+0][8] = 1  # k=3, i=0 -> index 9
    
    # Term r=9: A[0,2]*B[2,3]
    U[2][9] = 1
    V[2*5+3][9] = 1  # j=2, k=3 -> index 13
    W[3*n+0][9] = 1  # k=3, i=0 -> index 9
    
    # Term r=10: A[1,0]*B[0,3]
    U[3][10] = 1
    V[0*5+3][10] = 1  # j=0, k=3 -> index 3
    W[3*n+1][10] = 1  # k=3, i=1 -> index 10
    
    # Term r=11: A[1,1]*B[1,3]
    U[4][11] = 1
    V[1*5+3][11] = 1  # j=1, k=3 -> index 8
    W[3*n+1][11] = 1  # k=3, i=1 -> index 10
    
    # For k=4, use terms r=12..16
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Term r=12: A[2,0]*B[0,4]
    U[5][12] = 1
    V[0*5+4][12] = 1  # j=0, k=4 -> index 4
    W[4*n+0][12] = 1  # k=4, i=0 -> index 12
    
    # Term r=13: A[2,1]*B[1,4]
    U[6][13] = 1
    V[1*5+4][13] = 1  # j=1, k=4 -> index 9
    W[4*n+0][13] = 1  # k=4, i=0 -> index 12
    
    # Term r=14: A[1,2]*B[2,4]
    U[5][14] = 1
    V[2*5+4][14] = 1  # j=2, k=4 -> index 14
    W[4*n+1][14] = 1  # k=4, i=1 -> index 13
    
    # Term r=15: A[2,2]*B[2,4]
    U[8][15] = 1
    V[2*5+4][15] = 1  # j=2, k=4 -> index 14
    W[4*n+2][15] = 1  # k=4, i=2 -> index 14
    
    # We're missing some terms. Let me redo this more carefully.
    # This approach is getting complex. Let's verify what we have.
    
    # Actually, let's just return None and fall back to schoolbook for now.
    # The rank-17 construction needs more careful design.
    
    return None


def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all valid i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    
    return True
# EVOLVE-BLOCK-END
