import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _build_2x2x3_rank10():
    """
    Attempt to find a rank-10 decomposition for 2x2 matrix A and 2x3 matrix B.
    A is 2x2, B is 2x3, C is 2x3.
    Indices:
    A: 0,1,2,3 (i*2+j)
    B: 0,1,2,3,4,5 (j*3+k)
    C: 0,1,2,3,4,5 (k*2+i)
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This is a search problem. We can use a heuristic or a known decomposition.
    # For now, return None to indicate failure, triggering fallback.
    # In a real improvement, we would solve the system.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Rank 27 strategy via algebraic peeling
    rank_27 = _try_rank_27()
    if rank_27 is not None:
        return rank_27
        
    # Fallback to Rank 28
    return _strassen_rank28(n, m, p)

def _try_rank_27():
    """
    Apply algebraic peeling to the rank-28 decomposition to test if rank 27 is achievable.
    Strategy: Build the rank-28 decomposition, then try to find linear dependencies
    among the rank-1 terms that can be eliminated, leaving a rank-27 core.
    Then check if the residual error can be corrected with one additional rank-1 term.
    """
    n, m, p = 2, 3, 5
    rank28 = 28
    U28 = [[0] * rank28 for _ in range(n * m)]
    V28 = [[0] * rank28 for _ in range(m * p)]
    W28 = [[0] * rank28 for _ in range(p * n)]
    
    r = 0
    # Part 1: A0 B0 (j=0, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U28[i * m + 0][r] = 1
            V28[0 * p + k][r] = 1
            W28[k * n + i][r] = 1
            r += 1
            
    # Part 2: A12 B12 (j=1,2, k=0,1) - 7 terms (Strassen)
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        U28[i1*m+j1][r] = 1; U28[i2*m+j2][r] = 1
        V28[j1*p+k1][r] = 1; V28[j2*p+k2][r] = 1
        W28[k1*n+i1][r] = 1; W28[k2*n+i2][r] = 1
        r += 1
        U28[i2*m+j1][r] = 1; U28[i2*m+j2][r] = 1
        V28[j1*p+k1][r] = 1
        W28[k1*n+i2][r] = 1; W28[k2*n+i2][r] = -1
        r += 1
        U28[i1*m+j1][r] = 1
        V28[j1*p+k2][r] = 1; V28[j2*p+k2][r] = -1
        W28[k2*n+i1][r] = 1; W28[k2*n+i2][r] = 1
        r += 1
        U28[i2*m+j2][r] = 1
        V28[j2*p+k1][r] = 1; V28[j1*p+k1][r] = -1
        W28[k1*n+i1][r] = 1; W28[k1*n+i2][r] = 1
        r += 1
        U28[i1*m+j1][r] = 1; U28[i1*m+j2][r] = 1
        V28[j2*p+k2][r] = 1
        W28[k1*n+i1][r] = -1; W28[k2*n+i1][r] = 1
        r += 1
        U28[i2*m+j1][r] = 1; U28[i1*m+j1][r] = -1
        V28[j1*p+k1][r] = 1; V28[j1*p+k2][r] = 1
        W28[k2*n+i2][r] = 1
        r += 1
        U28[i1*m+j2][r] = 1; U28[i2*m+j2][r] = -1
        V28[j2*p+k1][r] = 1; V28[j2*p+k2][r] = 1
        W28[k1*n+i1][r] = 1
        r += 1

    add_strassen_2x2(0, 1, 1, 2, 0, 1)
    
    # Part 3: A12 B12 (j=1,2, k=2,3) - 7 terms (Strassen)
    add_strassen_2x2(0, 1, 1, 2, 2, 3)
    
    # Part 4: A12 B12 (j=1,2, k=4) - 4 terms (Direct)
    for i in range(n):
        for j in range(1, 3):
            U28[i * m + j][r] = 1
            V28[j * p + 4][r] = 1
            W28[4 * n + i][r] = 1
            r += 1
    
    # Now apply peeling: try to find if any rank-1 term is linearly dependent
    # on the others. For each term t, check if there exist coefficients c_s (s != t)
    # such that the residual tensor after removing term t and adding the correction
    # is zero.
    
    U28_arr = np.array(U28, dtype=float)
    V28_arr = np.array(V28, dtype=float)
    W28_arr = np.array(W28, dtype=float)
    
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to check if T can be represented with rank 27.
    
    # Peeling approach: For each candidate term t to remove, compute the residual
    # R[i,j,k] = T[i,j,k] - U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # Then check if R can be represented as a sum of 27 rank-1 terms from the remaining set.
    
    # A simpler approach: check if the 28 vectors (u_r, v_r, w_r) are linearly dependent
    # in some sense. For matrix multiplication tensors, the key invariant is that
    # the tensor must satisfy certain algebraic constraints.
    
    # Let's try a different peeling: check if we can express one term as a linear
    # combination of others in the tensor space.
    
    # For each term t, form the 3-way tensor contribution:
    # T_t[i,j,k] = U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    
    # Check if T_t is in the span of {T_s : s != t}
    # This is equivalent to checking if the matrix M_t (reshaped) is in the span.
    
    # Reshape each T_t into a matrix of size (n*m) x (m*p) by fixing k and varying i,j.
    # Actually, let's use a different approach: the tensor T_t can be viewed as a matrix
    # of size (n*m) x (p*m) by flattening (i,j) and (k).
    
    # For fixed k, T_t[i,j,k] as a function of (i,j) gives a vector in R^{n*m}.
    # Let's form the matrix A_t of size (n*m) x p where A_t[i*m+j, k] = U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] / W[k*n+i,t] if W != 0
    # This is getting complicated. Let's use a direct numerical approach.
    
    # Build the full tensor T as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.sum(U28_arr[i*m+j, :] * V28_arr[j*p+k, :] * W28_arr[k*n+i, :])
    
    # For each term t, compute the residual R_t = T - T_t
    # where T_t[i,j,k] = U28_arr[i*m+j, t] * V28_arr[j*p+k, t] * W28_arr[k*n+i, t]
    
    # Then check if R_t can be decomposed into rank 27 using the remaining terms.
    # This is equivalent to checking if the rank of the "flattened" residual is <= 27
    # in some appropriate sense.
    
    # A practical approach: for each t, try to solve for coefficients alpha_s (s != t)
    # such that T_t[i,j,k] = sum_{s != t} alpha_s * T_s[i,j,k]
    # This is a linear system in the alpha_s.
    
    # Flatten T_t and T_s into vectors of length n*m*p
    nmp = n * m * p
    T_flat = T.flatten()
    
    # For each t, build the matrix M where M[:, s] = T_s.flatten() for s != t
    # and solve M * alpha = T_t.flatten()
    
    best_t = -1
    best_alpha = None
    best_residual = float('inf')
    
    for t in range(rank28):
        T_t = np.zeros(nmp)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_t[i*m*p + j*p + k] = U28_arr[i*m+j, t] * V28_arr[j*p+k, t] * W28_arr[k*n+i, t]
        
        # Build matrix M with columns T_s for s != t
        idx = [s for s in range(rank28) if s != t]
        M = np.zeros((nmp, len(idx)))
        for col, s in enumerate(idx):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i*m*p + j*p + k, col] = U28_arr[i*m+j, s] * V28_arr[j*p+k, s] * W28_arr[k*n+i, s]
        
        # Solve M * alpha = T_t
        try:
            alpha, residuals, rank_M, sv = np.linalg.lstsq(M, T_t, rcond=None)
            residual = np.linalg.norm(M @ alpha - T_t)
            if residual < best_residual:
                best_residual = residual
                best_t = t
                best_alpha = alpha
        except:
            continue
    
    # If we found a term that is approximately in the span of the others,
    # we can remove it and the residual should be zero (or very small).
    # But this only works if the decomposition is not minimal.
    
    # For the matrix multiplication tensor, the rank is known to be 28 for 2x3x5
    # (or possibly 27). Let's check if the residual is close to zero.
    
    if best_t >= 0 and best_residual < 1e-6:
        # We found a dependent term. Remove it and check if the remaining 27 terms
        # still decompose the tensor.
        idx_keep = [s for s in range(rank28) if s != best_t]
        U27 = [[0] * 27 for _ in range(n * m)]
        V27 = [[0] * 27 for _ in range(m * p)]
        W27 = [[0] * 27 for _ in range(p * n)]
        
        for new_r, old_r in enumerate(idx_keep):
            for i in range(n):
                for j in range(m):
                    U27[i*m+j][new_r] = U28_arr[i*m+j, old_r]
            for j in range(m):
                for k in range(p):
                    V27[j*p+k][new_r] = V28_arr[j*p+k, old_r]
            for k in range(p):
                for i in range(n):
                    W27[k*n+i][new_r] = W28_arr[k*n+i, old_r]
        
        # Verify
        U27_arr = np.array(U27, dtype=float)
        V27_arr = np.array(V27, dtype=float)
        W27_arr = np.array(W27, dtype=float)
        
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U27_arr[i*m+j, :] * V27_arr[j*p+k, :] * W27_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U27, "V": V27, "W": W27}
    
    # If peeling didn't work directly, try a different approach:
    # Check if we can modify one term to absorb the dependency.
    # This is more complex and may not be feasible within time limits.
    
    return None

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A0 B0 (j=0, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A12 B12 (j=1,2, k=0,1) - 7 terms (Strassen)
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1
        W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
        r += 1
        U[i1*m+j1][r] = 1
        V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
        W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j2][r] = 1
        V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
        W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
        r += 1
        U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
        V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
        V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
        W[k2*n+i2][r] = 1
        r += 1
        U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
        V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1
        r += 1

    add_strassen_2x2(0, 1, 1, 2, 0, 1)
    
    # Part 3: A12 B12 (j=1,2, k=2,3) - 7 terms (Strassen)
    add_strassen_2x2(0, 1, 1, 2, 2, 3)
    
    # Part 4: A12 B12 (j=1,2, k=4) - 4 terms (Direct)
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
            
    if ok:
        return {"U": U, "V": V, "W": W}
        
    # If Strassen verification fails, try to repair or fall back
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
