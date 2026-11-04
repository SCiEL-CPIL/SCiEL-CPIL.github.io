"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition (rank 84)
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Try to reduce rank via rational elimination
        reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if reduced is not None:
            Ur, Vr, Wr, rrank = reduced
            if _verify_fast(Ur, Vr, Wr, n, m, p):
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        # Fallback to the 84-rank solution
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    Rank 84: 3 k-blocks * (2 Strassen + schoolbook for j=4 + bottom row)
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-84
    decomposition terms using rational arithmetic.
    
    The tensor equation is: for each (i,j,k) in {0..n-1}x{0..m-1}x{0..p-1},
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We have 84 terms. We check if any term can be expressed as a linear
    combination of others by examining the system of 90 equations.
    """
    r = U.shape[1]
    num_eq = n * m * p  # 90 equations
    if r <= num_eq:
        # Already at or below the number of equations, can't reduce further
        # without more sophisticated methods
        return None
    
    # Check time budget
    if time.time() > deadline - 20:
        return None
    
    # Build the system: for each (i,j,k), we have an equation
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is nonlinear in the factors, but we can check for linear
    # dependencies by looking at whether removing one term still allows
    # the system to be satisfied.
    
    # Strategy: try removing each term and check if the remaining terms
    # can still satisfy all 90 equations. This is expensive, so we limit
    # the search.
    
    # First, compute the tensor with all terms
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_full = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # For each term c, compute its contribution
    # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Try to see if T_c can be written as a linear combination of other T_c'
    
    # This is a complex nonlinear problem. Instead, let's try a simpler
    # approach: check if the 84 column-vectors (in the space of 90 equations)
    # have a linear dependency.
    
    # Actually, the tensor is bilinear in U and V for fixed W, etc.
    # Let's try a different approach: use the fact that the Strassen
    # decomposition has known structure and look for specific simplifications.
    
    # For the 3x5x6 case, the bottom row (i=2) uses schoolbook with rank 10
    # per k-block. The 1x5x2 sub-tensor has rank 5 (not 10) if we share
    # the W factors. Let's try to optimize the bottom row.
    
    # New approach: rebuild with optimized bottom row
    if time.time() > deadline - 10:
        return None
    
    return _build_optimized_3x5x6(n, m, p, deadline)


def _build_optimized_3x5x6(n, m, p, deadline):
    """
    Build an optimized decomposition with reduced rank by sharing
    factors in the bottom row computation.
    
    Key insight: For the bottom row (i=2), the 1x5x2 sub-tensor for each
    k-block can be done with rank 5 instead of 10 by sharing W factors.
    T[2,j,k] = sum_c U[10+j,c] * V[6j+k,c] * W[3k+2,c]
    
    With shared W: W[3k+2, c] = 1 for all j when using the same c.
    Then U[10+j,c] * V[6j+k,c] must sum to 1 for each (j,k).
    
    Actually, for 1x5x2: T[j,k] = 1 for all j,k.
    We can use rank 5: for c=0..4, set U[j,c]=delta(j,c), V[j*2+k,c]=delta(k,0)
    No wait, that doesn't work either.
    
    Let me think again. For 1x5x2 tensor T[j,k]=1:
    sum_c U[j,c] V[j*2+k,c] W[k,c] = 1 for all j in {0..4}, k in {0,1}
    
    With rank 5:
    c = 0..4
    U[j,c] = 1 if j==c else 0
    V[j*2+k,c] = 1 if j==c else 0  (so V[c*2+k, c] = 1 for k=0,1)
    W[k,c] = 1 for all k,c
    
    Check: sum_c U[j,c] V[j*2+k,c] W[k,c] = U[j,j] * V[j*2+k,j] * W[k,j] = 1*1*1 = 1 ✓
    
    So rank 5 works for 1x5x2!
    
    Similarly, for 1x1x2: T[0,k]=1 for k in {0,1}
    Rank 2: U[0,0]=1, U[0,1]=1, V[0,0]=1, V[1,0]=1, V[0,1]=1, V[1,1]=1, W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    Check: sum_c U[0,c] V[k,c] W[k,c]... 
    
    Actually for 1x1x2: n=1, m=1, p=2
    T[i,j,k] = 1 for i=0, j=0, k in {0,1}
    sum_c U[0*1+0, c] V[0*2+k, c] W[k*1+0, c] = 1
    = sum_c U[0,c] V[k,c] W[k,c] = 1
    
    With rank 2:
    c=0: U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    c=1: U[0,1]=0, ...
    
    Hmm, that gives sum = 1*1*1 + 1*1*1 = 2 for k=0. Not right.
    
    Let me use rank 2 differently:
    c=0: U[0,0]=1, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=0
    c=1: U[0,1]=1, V[0,1]=0, V[1,1]=1, W[0,1]=0, W[1,1]=1
    Check k=0: 1*1*1 + 1*0*0 = 1 ✓
    Check k=1: 1*0*0 + 1*1*1 = 1 ✓
    
    So 1x1x2 has rank 2.
    
    Revised rank calculation:
    Per k-block:
    - Top 2 rows, j in {0,1}: Strassen 2x2x2, rank 7
    - Top 2 rows, j in {2,3}: Strassen 2x2x2, rank 7
    - Top 2 rows, j=4: 2x1x2, rank 4
    - Bottom row, j in {0..4}: 1x5x2, rank 5 (shared W)
    Per k-block total: 7+7+4+5 = 23
    3 k-blocks: 69
    
    Wait, but the W factors for the bottom row need to be consistent across
    j values. Let me verify this works in the full context.
    
    In the full problem, for the bottom row:
    T[2,j,k] = sum_c U[10+j,c] V[6j+k,c] W[3k+2,c]
    
    For a fixed k-block (k0, k1), we need this for j=0..4, k in {k0,k1}.
    Using rank 5 for this sub-problem:
    For c_local = 0..4:
      U[10+c_local, c_global] = 1 (if c_global corresponds to this c_local)
      V[(6*j)+k, c_global] = 1 if j==c_local, else 0... 
    
    Wait, V depends on both j and k. V[j*6+k, c]. For the bottom row sub-problem,
    we need V[6j+k, c] to be nonzero only when j matches the "active" index.
    
    Let me re-examine. The 1x5x2 tensor has indices (j, k) with j in {0..4}, k in {0,1}.
    In the full problem, these map to:
    U index: 2*5 + j = 10 + j
    V index: j*6 + k (where k is the global k)
    W index: k*3 + 2
    
    For a k-block with global k values {k0, k1}:
    We need: sum_c U[10+j, c] * V[6j+k, c] * W[3k+2, c] = 1 for j=0..4, k in {k0,k1}
    
    With 5 terms (c = c0..c4):
    U[10+j, c0+j] = 1 for j=0..4 (i.e., U[10, c0]=1, U[11, c0+1]=1, ..., U[14, c0+4]=1)
    V[6j+k, c0+j] = 1 for all j=0..4, k in {k0,k1}
    W[3k+2, c0+j] = 1 for all k in {k0,k1}, j=0..4
    
    Check: sum_c U[10+j, c] V[6j+k, c] W[3k+2, c]
    = U[10+j, c0+j] * V[6j+k, c0+j] * W[3k+2, c0+j]
    = 1 * 1 * 1 = 1 ✓
    
    Great! So the bottom row per k-block uses rank 5 instead of 10.
    
    New total rank: 3 * (7 + 7 + 4 + 5) = 3 * 23 = 69
    
    Let me also check if we can reduce the 2x1x2 block (top rows, j=4).
    2x1x2 tensor: T[i, 0, k] = 1 for i in {0,1}, k in {0,1}
    In the full problem:
    U[i*5+4, c] * V[4*6+k, c] * W[k*3+i, c] = 1
    
    With rank 4 (schoolbook): each (i,k) pair gets its own term.
    Can we do better? 
    c=0: U[4,0]=1, U[9,0]=1, V[24,0]=1, V[25,0]=1, W[0,0]=1, W[1,0]=1, W[3,0]=1, W[4,0]=1
    This gives sum = 1*1*1 + 1*1*1 = 2 for (i=0,k=0). Not right.
    
    Let me try rank 2:
    c=0: U[4,0]=1, U[9,0]=0, V[24,0]=1, V[25,0]=0, W[0,0]=1, W[1,0]=0, W[3,0]=0, W[4,0]=1
    c=1: U[4,1]=0, U[9,1]=1, V[24,1]=0, V[25,1]=1, W[0,1]=0, W[1,1]=1, W[3,1]=1, W[4,1]=0
    
    Check (i=0, k=0): U[4,0]*V[24,0]*W[0,0] + U[4,1]*V[24,1]*W[0,1] = 1*1*1 + 0 = 1 ✓
    Check (i=0, k=1): U[4,0]*V[25,0]*W[3,0] + U[4,1]*V[25,1]*W[3,1] = 1*0*0 + 0 = 0 ✗
    
    That doesn't work. The issue is that V[25,0] should be 1 for (i=0,k=1) but we set it to 0.
    
    Actually, for 2x1x2, the minimum rank is 4 (it's a 2x2 matrix of ones, and the
    tensor rank of a 2x1x2 all-ones tensor is 4 since it's equivalent to a 2x2 matrix
    which has rank 4 as a tensor... no wait.
    
    T[i,0,k] = 1 for i in {0,1}, k in {0,1}. This is a 2x2 matrix of ones.
    The tensor rank of the 2x2 all-ones matrix (as a 2x1x2 tensor) is...
    Actually, a 2x2 matrix M has tensor rank equal to its matrix rank if we view
    it as a 2x1x2 tensor? No, that's not right either.
    
    Let me just use rank 4 for 2x1x2 (schoolbook).
    
    So total rank = 3 * (7 + 7 + 4 + 5) = 69.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Per k-block: 7 (Strassen j=0,1) + 7 (Strassen j=2,3) + 4 (schoolbook j=4 top) + 5 (optimized bottom) = 23
    rank = 3 * 23
    U = np.zeros((n * m, rank))  # 15 x 69
    V = np.zeros((m * p, rank))  # 30 x 69
    W = np.zeros((p * n, rank))  # 18 x 69
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2 (rank 4)
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0..4}: optimized 1x5x2 (rank 5)
        for j in range(5):
            U[2*5 + j, col] = 1
            V[j*6 + k0, col] = 1
            V[j*6 + k1, col] = 1
            W[k0*3 + 2, col] = 1
            W[k1*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
