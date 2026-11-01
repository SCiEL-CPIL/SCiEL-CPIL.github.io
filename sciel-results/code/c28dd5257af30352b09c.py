"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-6 exact tensor decomposition for 2x2 matrix multiplication
    over F_3. If no solution exists, fall back to the verified rank-7 Strassen decomposition.
    
    The matrix multiplication tensor for 2x2 matrices is a 3-tensor T with indices
    (i, a, k) where i is row of A, a is column of A / row of B, k is column of B.
    T[i, a, k] = 1 for all i, a, k in {0, 1}.
    
    A rank-r decomposition writes:
    T[i, a, k] = sum_{r=1}^{r} U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]
    
    where:
    - U has shape (n*m, rank) = (4, rank)
    - V has shape (m*p, rank) = (4, rank)
    - W has shape (p*n, rank) = (4, rank)
    
    For rank 6 over F_3, we search for entries in {0, 1, 2} (mod 3).
    However, it is a well-known result that the tensor rank of 2x2 matrix multiplication
    is exactly 7 (over any field of characteristic not 2). Therefore, no rank-6 solution
    exists, and we return the Strassen rank-7 decomposition.
    """
    n, m, p = 2, 2, 2
    
    # First, attempt rank-6 search over F_3 (will always fail, but demonstrates the approach)
    # For efficiency, we skip the exhaustive search since we know rank 6 is impossible.
    # The search would involve solving a system of polynomial equations over F_3.
    
    # Fall back to verified rank-7 Strassen decomposition
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # M1: A part: A00 + A11, B part: B00 + B11
    # U[i*m + a, r] where i is row of A, a is col of A
    U[0*2+0][0] = 1  # A[0,0]
    U[1*2+1][0] = 1  # A[1,1]
    # V[a*p + k, r] where a is row of B, k is col of B
    V[0*2+0][0] = 1  # B[0,0]
    V[1*2+1][0] = 1  # B[1,1]
    
    # M2: A part: A10 + A11, B part: B00
    U[1*2+0][1] = 1  # A[1,0]
    U[1*2+1][1] = 1  # A[1,1]
    V[0*2+0][1] = 1  # B[0,0]
    
    # M3: A part: A00, B part: B01 - B11
    U[0*2+0][2] = 1  # A[0,0]
    V[0*2+1][2] = 1  # B[0,1]
    V[1*2+1][2] = -1 # -B[1,1]
    
    # M4: A part: A11, B part: B10 - B00
    U[1*2+1][3] = 1  # A[1,1]
    V[1*2+0][3] = 1  # B[1,0]
    V[0*2+0][3] = -1 # -B[0,0]
    
    # M5: A part: A00 + A01, B part: B11
    U[0*2+0][4] = 1  # A[0,0]
    U[0*2+1][4] = 1  # A[0,1]
    V[1*2+1][4] = 1  # B[1,1]
    
    # M6: A part: A10 - A00, B part: B00 + B01
    U[1*2+0][5] = 1  # A[1,0]
    U[0*2+0][5] = -1 # -A[0,0]
    V[0*2+0][5] = 1  # B[0,0]
    V[0*2+1][5] = 1  # B[0,1]
    
    # M7: A part: A01 - A11, B part: B10 + B11
    U[0*2+1][6] = 1  # A[0,1]
    U[1*2+1][6] = -1 # -A[1,1]
    V[1*2+0][6] = 1  # B[1,0]
    V[1*2+1][6] = 1  # B[1,1]
    
    # W[k*n + i, r] where k is col of C, i is row of C
    # C[0,0] = M1 + M4 - M5 + M7
    W[0*2+0][0] = 1   # M1
    W[0*2+0][3] = 1   # M4
    W[0*2+0][4] = -1  # -M5
    W[0*2+0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[1*2+0][2] = 1   # M3
    W[1*2+0][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[0*2+1][1] = 1   # M2
    W[0*2+1][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[1*2+1][0] = 1   # M1
    W[1*2+1][1] = -1  # -M2
    W[1*2+1][2] = 1   # M3
    W[1*2+1][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
