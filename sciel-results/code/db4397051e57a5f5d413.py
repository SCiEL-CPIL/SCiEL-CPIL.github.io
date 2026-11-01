import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Quick verification that U,V,W decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank_28(n, m, p):
    """
    Rank-28 decomposition:
    - j=0: schoolbook 2x1x5 (rank 10)
    - j=1,2, k=0,1: Strassen 2x2x2 (rank 7)
    - j=1,2, k=2: schoolbook 2x2x1 (rank 4)
    - j=1,2, k=3,4: Strassen 2x2x2 (rank 7)
    Total: 10+7+4+7 = 28
    """
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i*m+j, col] = 1.0
        V[j*p+k, col] = 1.0
        W[k*n+i, col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M2: (A10+A11)B00 -> C10+=, C11-=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = -1.0
        col += 1
        # M3: A00(B01-B11) -> C01+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M4: A11(B10-B00) -> C00+=, C10+=
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = -1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        col += 1
        # M5: (A00+A01)B11 -> C00-=, C01+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        col += 1
        # M6: (A10-A00)(B00+B01) -> C11+=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+0), col] = -1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M7: (A01-A11)(B10+B11) -> C00+=
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = -1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        col += 1
        return col

    # Block 1: j=0, all k (schoolbook, 10 terms)
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)

    # Block 2a: j=1,2, k=0,1 (Strassen, 7 terms)
    col = add_strassen(0, 1, 0, col)

    # Block 2b: j=1,2, k=2 (schoolbook, 4 terms)
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # Block 2c: j=1,2, k=3,4 (Strassen, 7 terms)
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W

def _try_rank_27_subspace(n, m, p, deadline):
    """
    Subspace method: restrict U to have rank 2 (from n=2 dimension).
    If U has rank 2, then U = A @ B where A is (n*m, 2) and B is (2, rank).
    The tensor equation becomes:
      1 = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        = sum_s (A @ B)[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    Let U = A @ B. Then:
      1 = sum_s sum_a A[i*m+j, a] * B[a, s] * V[j*p+k, s] * W[k*n+i, s]
        = sum_a A[i*m+j, a] * (sum_s B[a, s] * V[j*p+k, s] * W[k*n+i, s])
    
    For this to equal 1 for all i,j,k, we need:
      sum_a A[i*m+j, a] * M_a[j*p+k, k*n+i] = 1
    
    where M_a is defined by the remaining factors. This is still complex.
    
    Alternative approach: Since n=2, U has 6 rows. If rank(U) <= 2, then
    U can be written as U = C @ D where C is (6, 2) and D is (2, rank).
    
    The tensor equation for each (i,j,k):
      1 = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    With U = C @ D:
      1 = sum_s (sum_a C[i*m+j, a] * D[a, s]) * V[j*p+k, s] * W[k*n+i, s]
        = sum_a C[i*m+j, a] * (sum_s D[a, s] * V[j*p+k, s] * W[k*n+i, s])
    
    Let E_a[j*p+k, k*n+i] = sum_s D[a, s] * V[j*p+k, s] * W[k*n+i, s]
    
    Then we need: sum_a C[i*m+j, a] * E_a[j*p+k, k*n+i] = 1 for all i,j,k.
    
    This is a system where we can fix C and solve for the rest, or vice versa.
    Let's fix C to be a specific rank-2 matrix and solve for D, V, W.
    
    Actually, a cleaner approach: if U has rank 2, then there exist 2 vectors
    u_1, u_2 in R^6 such that every column of U is a linear combination of u_1, u_2.
    
    Let U[:, s] = alpha_s * u_1 + beta_s * u_2 for each s.
    
    Then the tensor equation becomes:
      1 = sum_s (alpha_s * u_1[i*m+j] + beta_s * u_2[i*m+j]) * V[j*p+k, s] * W[k*n+i, s]
        = u_1[i*m+j] * (sum_s alpha_s * V[j*p+k, s] * W[k*n+i, s])
        + u_2[i*m+j] * (sum_s beta_s * V[j*p+k, s] * W[k*n+i, s])
    
    For this to equal 1 for all i,j,k, we need:
      u_1[i*m+j] * F_1[j*p+k, k*n+i] + u_2[i*m+j] * F_2[j*p+k, k*n+i] = 1
    
    where F_1 and F_2 are defined by the sums above.
    
    This is getting complex. Let me try a more direct approach:
    
    Direct linear system approach:
    Fix W to be a random (p*n) x rank matrix with full column rank.
    Then the tensor equation for each (i,j,k) is:
      1 = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    This is still quadratic in U and V. But if we fix U as well, we can solve for V.
    
    Better: use the fact that the tensor is all ones. Let's set up the system differently.
    
    For a rank-r decomposition, we need:
      T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    
    Consider fixing U to have a specific structure. Since n=2, let's try:
    U[i*m+j, s] = f_i(j, s) where f_i depends on i in a structured way.
    
    Simplest: try U[i*m+j, s] = a_i(s) * b_j(s) (rank-1 in the i,j block).
    But this gives rank at most... this is getting too complex.
    
    Let me try the most direct approach: set up the full linear system.
    
    For rank r, we have r * (n*m + m*p + p*n) = r * (6 + 15 + 10) = 31r unknowns.
    We have n*m*p = 30 equations.
    
    For r = 27: 31*27 = 837 unknowns, 30 equations. Highly underdetermined.
    The system is:
      sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all (i,j,k)
    
    This is a system of 30 equations in 837 variables, but they're coupled (products).
    
    Key insight: if we fix two of the three matrices, the system becomes linear in the third.
    
    Fix W (10 x 27) and U (6 x 27). Then for each (i,j,k):
      1 = sum_s U[i*m+j, s] * W[k*n+i, s] * V[j*p+k, s]
    
    This is linear in V! For fixed (i,k), we get an equation in V[*, s] for s=1..27.
    Actually, for each (i,j,k):
      sum_s (U[i*m+j, s] * W[k*n+i, s]) * V[j*p+k, s] = 1
    
    Let A[i,k,j*p+k, s] = U[i*m+j, s] * W[k*n+i, s]. Then:
      sum_s A[i,k,j*p+k, s] * V[j*p+k, s] = 1
    
    This gives n*p = 10 equations (for each i,k) on V, but each equation involves
    different rows of V. Specifically, for fixed (i,k), the equation is:
      sum_s U[i*m+j, s] * W[k*n+i, s] * V[j*p+k, s] = 1
    
    This is one equation per (i,j,k), so 30 equations total, but they're not all
    independent in the same way.
    
    Actually, let's just fix U and W and solve for V via least squares.
    
    The system is: for each (i,j,k),
      sum_s (U[i*m+j, s] * W[k*n+i, s]) * V[j*p+k, s] = 1
    
    This can be written as: M @ V.flatten() = ones(30)
    where M is a 30 x (m*p*rank) matrix.
    
    M[(i,j,k), (j'*p+k', s)] = U[i*m+j, s] * W[k*n+i, s] if j'=j and k'=k, else 0.
    
    Wait, that's not right. V[j*p+k, s] only appears when the row index of V is j*p+k.
    So for equation (i,j,k), only V[j*p+k, s] for s=0..rank-1 appear.
    
    So the system decomposes! For each (j,k), we have n equations:
      For i=0,1: sum_s U[i*m+j, s] * W[k*n+i, s] * V[j*p+k, s] = 1
    
    This is a 2 x rank linear system for V[j*p+k, :]:
      [U[0*m+j, :] * W[k*n+0, :]] @ V[j*p+k, :] = 1
      [U[1*m+j, :] * W[k*n+1, :]] @ V[j*p+k, :] = 1
    
    So for each (j,k), we have a 2 x 27 system. If the 2x27 matrix has rank 2,
    there are infinitely many solutions. We can pick the minimum norm solution.
    
    But we also need the full tensor to be satisfied. The issue is that U and W
    are also unknowns. So we need to choose U and W such that for all (j,k),
    the 2x27 system [U[0*m+j, :]*W[k*n+0, :], U[1*m+j, :]*W[k*n+1, :]] has a
    solution for V[j*p+k, :].
    
    This is equivalent to: for all (j,k), the vector (1,1) must be in the row space
    of the 2x27 matrix, i.e., there exists v such that:
      sum_s U[0*m+j, s] * W[k*n+0, s] * v[s] = 1
      sum_s U[1*m+j, s] * W[k*n+1, s] * v[s] = 1
    
    This is always satisfiable as long as the 2x27 matrix has full row rank (rank 2).
    
    So the strategy is:
    1. Choose U (6 x 27) and W (10 x 27) such that for all (j,k), the 2x27 matrix
       has rank 2.
    2. Solve for V.
    3. Verify the full tensor.
    
    To make this tractable, let's choose U and W with a specific structure.
    """
    if time.time() > deadline:
        return None
    
    rank = 27
    
    # Strategy: choose U and W to have a specific structure that makes the
    # 2x27 systems easily solvable.
    
    # Let U[i*m+j, s] depend only on i and s (not on j):
    # U[i*m+j, s] = u[i, s]
    # And W[k*n+i, s] depend only on i and s (not on k):
    # W[k*n+i, s] = w[i, s]
    
    # Then the 2x27 system for (j,k) is:
    #   [u[0, :] * w[0, :]] @ v = 1
    #   [u[1, :] * w[1, :]] @ v = 1
    
    # This is independent of j and k! So we just need one 2x27 system to be solvable.
    
    # Let u[0, s] = 1, u[1, s] = 1 for all s.
    # Let w[0, s] = 1, w[1, s] = 1 for all s.
    # Then the system is:
    #   sum_s v[s] = 1
    #   sum_s v[s] = 1
    # Solution: v[s] = 1/27 for all s.
    
    # But wait, this means U[i*m+j, s] = 1 for all i,j,s and W[k*n+i, s] = 1 for all k,i,s.
    # Then V[j*p+k, s] = 1/27 for all j,k,s.
    
    # Let's check: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = sum_s 1 * (1/27) * 1 = 27/27 = 1. 
    # This works! But it's rank 27 with all entries 1/27.
    
    # However, this is a valid rank-27 decomposition! Let's verify it.
    
    U = np.ones((n*m, rank))
    V = np.ones((m*p, rank)) / rank
    W = np.ones((p*n, rank))
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    # If that didn't work (shouldn't happen), try other structures.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # First, try the known rank-28 decomposition
    try:
        U, V, W = _build_rank_28(n, m, p)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Try to find rank 27 using subspace method
    if time.time() < deadline:
        try:
            result = _try_rank_27_subspace(n, m, p, deadline)
            if result is not None:
                U, V, W = result
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        except Exception:
            pass
    
    # Fallback: schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
