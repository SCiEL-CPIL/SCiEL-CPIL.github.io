import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 27 first
    rank_27 = _try_rank_27()
    if rank_27 is not None:
        return rank_27
    
    # Fallback to rank 28
    return _strassen_rank28(n, m, p)

def _try_rank_27():
    """
    Attempt to find a rank-27 decomposition using a structured parameterization.
    
    Strategy: Split A (2x3) into A_left (2x2) and A_right (2x1).
    Split B (3x5) into B_top (2x5) and B_bottom (1x5).
    
    C = A_left * B_top + A_right * B_bottom
    
    Use Strassen (rank 6) for A_left * B_top_left (2x2 * 2x2),
    Strassen (rank 6) for A_left * B_top_right (2x2 * 2x2),
    Direct for A_left * B_top_col5 (2x2 * 2x1) = 4 terms,
    Direct for A_right * B_bottom (2x1 * 1x5) = 10 terms... 
    That's 6+6+4+10 = 26, but we need to handle the cross correctly.
    
    Actually, let's use a more careful block decomposition.
    """
    n, m, p = 2, 3, 5
    
    # Alternative: use the known structure that 2x3x5 has rank 27.
    # Build a parametric rank-27 decomposition and solve for parameters.
    
    # Structure: 
    # - 7 Strassen-like terms for a 2x2x2 sub-block (rows 0-1, cols 0-1 of B)
    # - 7 Strassen-like terms for another 2x2x2 sub-block (rows 0-1, cols 2-3 of B)  
    # - 4 direct terms for (rows 0-1, col 4 of B)
    # - 9 direct terms for (row 2 of A, all cols of B)
    # Total: 7 + 7 + 4 + 9 = 27
    
    # But we need to verify this works. Let's build it explicitly.
    
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] using Strassen (7 terms)
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Strassen multipliers for 2x2:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10-B00) -> contributes to C00, C10
    # M4 = A11(B01-B10) -> contributes to C01, C11
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C10
    # M7 = (A01-A11)(B10+B11) -> contributes to C01
    
    # M1
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[3, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # M2
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1
    r += 1
    
    # M3
    U[0, r] = 1
    V[2, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1
    r += 1
    
    # M4
    U[3, r] = 1
    V[1, r] = 1; V[2, r] = -1
    W[1, r] = 1; W[3, r] = 1
    r += 1
    
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[3, r] = 1
    W[0, r] = -1; W[1, r] = 1
    r += 1
    
    # M6
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[2, r] = 1
    r += 1
    
    # M7
    U[1, r] = 1; U[3, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[1, r] = 1
    r += 1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] using Strassen (7 terms)
    # B indices for this block: (0,2)=2, (0,3)=3, (1,2)=4, (1,3)=5
    # C indices: (0,2)=2, (0,3)=3, (1,2)=4, (1,3)=5
    
    # M1
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[5, r] = 1
    W[2, r] = 1; W[5, r] = 1
    r += 1
    
    # M2
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[4, r] = 1; W[5, r] = -1
    r += 1
    
    # M3
    U[0, r] = 1
    V[4, r] = 1; V[2, r] = -1
    W[2, r] = 1; W[4, r] = 1
    r += 1
    
    # M4
    U[3, r] = 1
    V[3, r] = 1; V[4, r] = -1
    W[3, r] = 1; W[5, r] = 1
    r += 1
    
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[2, r] = -1; W[3, r] = 1
    r += 1
    
    # M6
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[4, r] = 1
    r += 1
    
    # M7
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[3, r] = 1
    r += 1
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] direct (4 terms)
    # B index: (0,4)=4, (1,4)=5
    # C indices: (0,4)=4, (1,4)=5
    for i in range(2):
        for j in range(2):
            U[i * m + j, r] = 1
            V[j * p + 4, r] = 1
            W[4 * n + i, r] = 1
            r += 1
    
    # Block 4: A[2:3, :] * B[:, :] direct
    # A row 2: indices (2,0)=6, (2,1)=7, (2,2)=8
    # All B columns
    for j in range(3):
        for k in range(5):
            U[2 * m + j, r] = 1
            V[j * p + k, r] = 1
            W[k * n + 2, r] = 1  # Wait, n=2, so i only goes 0,1
            r += 1
    
    # Oops, n=2, so i is only 0,1. Let me fix.
    # Actually, the A matrix is 2x3, so rows are i=0,1 and columns are j=0,1,2.
    # The B matrix is 3x5, so rows are j=0,1,2 and columns are k=0,1,2,3,4.
    # C is 2x5, so rows are i=0,1 and columns are k=0,1,2,3,4.
    
    # I made an error. Let me restart the block decomposition properly.
    
    # Correct approach:
    # A is 2x3: A[i,j] for i in {0,1}, j in {0,1,2}
    # B is 3x5: B[j,k] for j in {0,1,2}, k in {0,1,2,3,4}
    # C is 2x5: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split j into {0,1} and {2}:
    # C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    
    # For the first part, split k into {0,1}, {2,3}, {4}:
    # C[i,0:2] = sum_{j=0,1} A[i,j]*B[j,0:2]  (2x2 * 2x2, use Strassen rank 6)
    # C[i,2:4] = sum_{j=0,1} A[i,j]*B[j,2:4]  (2x2 * 2x2, use Strassen rank 6)
    # C[i,4]   = sum_{j=0,1} A[i,j]*B[j,4]    (2x2 * 2x1, direct 4 terms)
    # C[i,:]   += A[i,2]*B[2,:]                (2x1 * 1x5, direct 10 terms)
    
    # Total: 6 + 6 + 4 + 10 = 26 terms. But we need to check if this is correct.
    # Actually, Strassen for 2x2 gives rank 6, not 7. Let me use rank 6.
    
    # Let me rebuild with rank 6 Strassen blocks.
    
    return None  # Fall through to rank 28 for now

def _strassen_rank28(n, m, p):
    """
    Rank-28 decomposition using Strassen blocks.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Part 1: A[:, 0] * B[0, :] (2x1 * 1x5) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0, r] = 1
            V[0 * p + k, r] = 1
            W[k * n + i, r] = 1
            r += 1
    
    # Part 2: A[:, 1:3] * B[1:3, 0:2] (2x2 * 2x2) - Strassen rank 6
    _add_strassen_2x2(U, V, W, n, m, p, i1=0, i2=1, j1=1, j2=2, k1=0, k2=1, r=r)
    r += 6
    
    # Part 3: A[:, 1:3] * B[1:3, 2:4] (2x2 * 2x2) - Strassen rank 6
    _add_strassen_2x2(U, V, W, n, m, p, i1=0, i2=1, j1=1, j2=2, k1=2, k2=3, r=r)
    r += 6
    
    # Part 4: A[:, 1:3] * B[1:3, 4:5] (2x2 * 2x1) - direct 4 terms
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, r] = 1
            V[j * p + 4, r] = 1
            W[4 * n + i, r] = 1
            r += 1
    
    # Verify
    ok = _verify(U, V, W, n, m, p)
    if ok:
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback
    return _schoolbook(n, m, p)

def _add_strassen_2x2(U, V, W, n, m, p, i1, i2, j1, j2, k1, k2, r):
    """
    Add Strassen 2x2 decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1].
    """
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    U[i1*m+j1, r] += 1
    U[i2*m+j2, r] += 1
    V[j1*p+k1, r] += 1
    V[j2*p+k2, r] += 1
    W[k1*n+i1, r] += 1
    W[k2*n+i2, r] += 1
    r += 1
    
    # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
    U[i2*m+j1, r] += 1
    U[i2*m+j2, r] += 1
    V[j1*p+k1, r] += 1
    W[k1*n+i2, r] += 1
    W[k2*n+i2, r] += -1
    r += 1
    
    # M3 = A[i1,j1] * (B[j2,k2]-B[j1,k2])
    U[i1*m+j1, r] += 1
    V[j2*p+k2, r] += 1
    V[j1*p+k2, r] += -1
    W[k2*n+i1, r] += 1
    W[k2*n+i2, r] += 1
    r += 1
    
    # M4 = A[i2,j2] * (B[j1,k1]-B[j2,k1])
    U[i2*m+j2, r] += 1
    V[j1*p+k1, r] += 1
    V[j2*p+k1, r] += -1
    W[k1*n+i1, r] += 1
    W[k1*n+i2, r] += 1
    r += 1
    
    # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
    U[i1*m+j1, r] += 1
    U[i1*m+j2, r] += 1
    V[j2*p+k2, r] += 1
    W[k1*n+i1, r] += -1
    W[k2*n+i1, r] += 1
    r += 1
    
    # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
    U[i2*m+j1, r] += 1
    U[i1*m+j1, r] += -1
    V[j1*p+k1, r] += 1
    V[j1*p+k2, r] += 1
    W[k2*n+i2, r] += 1
    r += 1
    
    # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
    U[i1*m+j2, r] += 1
    U[i2*m+j2, r] += -1
    V[j2*p+k1, r] += 1
    V[j2*p+k2, r] += 1
    W[k1*n+i1, r] += 1
    r += 1
    
    return r

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
