"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U_f[i*m+j])):
                    total += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                if total != 1:
                    return False
    return True

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

def _construct_rank50(n, m, p):
    """
    Construct a rank-50 decomposition for 2x3x10.
    Uses 5 terms per output column k (rank of 2x3x1 is 5).
    
    For C[i,k] = sum_j A[i,j]*B[j,k]:
    We need 5 multiplications per k.
    Standard rank-5 decomposition of 2x3x1:
      M0 = A[0,0]*B[0]
      M1 = A[0,1]*B[1]  
      M2 = A[0,2]*B[2]
      M3 = (A[1,0]-A[0,0])*B[0]
      M4 = (A[1,1]-A[0,1])*B[1]
    Then:
      C[0,0] = M0
      C[0,1] = M1
      C[0,2] = M2
      C[1,0] = M0 + M3
      C[1,1] = M1 + M4
      C[1,2] = M2 + (A[1,2]-A[0,2])*B[2]  -- wait, this needs another term!
    
    Actually for 2x3x1, the rank is 5, but we need to be careful.
    Let me use a known rank-5 scheme:
    
    For multiplying 2x3 by 3x1:
    We can use:
      M0 = A[0,0]*B[0]
      M1 = A[0,1]*B[1]
      M2 = A[0,2]*B[2]
      M3 = (A[1,0]+A[0,0])*B[0]
      M4 = (A[1,1]+A[0,1])*B[1]
    
    Then:
      C[0,0] = M0
      C[0,1] = M1
      C[0,2] = M2
      C[1,0] = M3 - M0
      C[1,1] = M4 - M1
      C[1,2] = ? We still need A[1,2]*B[2]
    
    Hmm, 5 terms isn't enough for 2x3x1 if we need all 6 products.
    Actually the tensor rank of 2x3x1 matrix multiplication is 5.
    Let me look up the correct scheme.
    
    The correct rank-5 decomposition of 2x3x1:
    M0 = A[0,0]*B[0]
    M1 = A[0,1]*B[1]
    M2 = A[0,2]*B[2]
    M3 = (A[1,0]-A[0,0])*B[0]
    M4 = (A[1,1]-A[0,1])*B[1]
    
    C[0,0] = M0
    C[0,1] = M1
    C[0,2] = M2
    C[1,0] = M0 + M3
    C[1,1] = M1 + M4
    C[1,2] = M2 + (A[1,2]-A[0,2])*B[2]
    
    But (A[1,2]-A[0,2])*B[2] is not one of M0-M4!
    So rank 5 doesn't work for 2x3x1 with this scheme.
    
    Actually, I recall that the rank of 2x3x1 matrix multiplication tensor is 5.
    Let me use a different scheme:
    
    M0 = A[0,0]*B[0]
    M1 = A[0,1]*B[1]
    M2 = A[0,2]*B[2]
    M3 = (A[1,0]+A[0,1])*B[0]
    M4 = (A[1,1]+A[0,2])*B[1]
    
    This is getting complicated. Let me just use rank 6 per k (schoolbook for 2x3x1)
    giving total rank 60, which is already better than 60 (schoolbook).
    
    Wait, schoolbook is 2*3*10 = 60. So 6 per k is 60 total, same as schoolbook.
    
    Let me try a different approach: use Strassen-like ideas for the 3x10 part.
    For 3x10, we can group columns: 3x10 = 3x(3+3+3+1) or similar.
    Actually, let's use the fact that we're multiplying by a 3x10 matrix.
    We can view this as 10 separate 3x1 multiplications, each with rank 3 (schoolbook).
    Total = 30. But we can do better by sharing.
    
    Let me try a direct construction with rank 45 or so.
    """
    # For now, return None to indicate this needs more work
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p  # 60
    
    # Try to find a better decomposition
    # Strategy: Use the structure of 2x3x10 to find a lower-rank decomposition
    
    # Approach: For each pair (i,k) where i in {0,1} and k in {0..9},
    # we need C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Key insight: We can use a Strassen-like decomposition.
    # For 2x3x1, the tensor rank is 5 (known result).
    # Let me verify this and construct it properly.
    
    # Rank-5 decomposition of 2x3x1 matrix multiplication:
    # Reference: This is a standard result.
    # 
    # M0 = A[0,0] * B[0]
    # M1 = A[0,1] * B[1]
    # M2 = A[0,2] * B[2]
    # M3 = (A[1,0] + A[0,1]) * B[0]
    # M4 = (A[1,1] + A[0,2]) * B[1]
    #
    # C[0,0] = M0
    # C[0,1] = M1
    # C[0,2] = M2
    # C[1,0] = M3 - M0 - M1  ... this doesn't work cleanly
    
    # Let me use a verified rank-5 scheme:
    # Actually, let me just construct a rank-45 decomposition directly.
    # Use 5 terms per k, but with a valid scheme.
    
    # Valid rank-5 scheme for 2x3x1 (verified):
    # M0 = A[0,0]*B[0]
    # M1 = A[0,1]*B[1]
    # M2 = A[0,2]*B[2]
    # M3 = (A[1,0]-A[0,0])*B[0]
    # M4 = (A[1,2]-A[0,2])*B[2]
    #
    # C[0,0] = M0
    # C[0,1] = M1
    # C[0,2] = M2
    # C[1,0] = M0 + M3
    # C[1,1] = M1  ... wait, C[1,1] should be A[1,1]*B[1]
    # This doesn't give C[1,1] correctly.
    
    # I'll use a different approach: just use rank 6 per k (schoolbook for each 2x3x1)
    # Total rank = 60, same as overall schoolbook. Not an improvement.
    
    # Let me try to find a true improvement using optimization.
    # For now, return schoolbook.
    
    return best_result
# EVOLVE-BLOCK-END
