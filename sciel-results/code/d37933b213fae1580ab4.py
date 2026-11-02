"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _combine_decompositions(n, m1, p1, m2, p2, U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions:
      C_left  = A[:,:m1] @ B[:m1,:]   using (U1, V1, W1)
      C_right = A[:,m1:] @ B[m1:,:]   using (U2, V2, W2)
    Result: C = C_left + C_right
    
    U has rows indexed by (i, j) where i in [0,n), j in [0, m1+m2)
    V has rows indexed by (j, k) where j in [0, m1+m2), k in [0, p1+p2)
    W has rows indexed by (k, i) where k in [0, p1+p2), i in [0,n)
    """
    m = m1 + m2
    p = p1 + p2
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left part: j in [0, m1), k in [0, p1)
    for i in range(n):
        for j in range(m1):
            for r in range(rank1):
                U[i * m + j][r] = U1[i * m1 + j][r]
    for j in range(m1):
        for k in range(p1):
            for r in range(rank1):
                V[j * p + k][r] = V1[j * p1 + k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(rank1):
                W[k * n + i][r] = W1[k * n + i][r]
    
    # Right part: j in [m1, m), k in [p1, p)
    for i in range(n):
        for j in range(m1, m):
            j2 = j - m1
            for r in range(rank2):
                U[i * m + j][rank1 + r] = U2[i * m2 + j2][r]
    for j in range(m1, m):
        for k in range(p1, p):
            j2 = j - m1
            k2 = k - p1
            for r in range(rank2):
                V[j * p + k][rank1 + r] = V2[j2 * p2 + k2][r]
    for k in range(p1, p):
        for i in range(n):
            k2 = k - p1
            for r in range(rank2):
                W[k * n + i][rank1 + r] = W2[k2 * n + i][r]
    
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 multiplications for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-c)(e+g)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i*m+j][r] where i=0,1 (rows of A), j=0,1 (cols of A)
    # V[j*p+k][r] where j=0,1 (rows of B), k=0,1 (cols of B)
    # W[k*n+i][r] where k=0,1 (cols of B), i=0,1 (rows of A)
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # r=0: M1 = (a+d)(e+h)
    # a = U[0*2+0]=U[0], d = U[1*2+1]=U[3]
    # e = V[0*2+0]=V[0], h = V[1*2+1]=V[3]
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0*2+0][0] = 1  # C11 += M1
    W[1*2+1][0] = 1  # C22 += M1
    
    # r=1: M2 = (c+d)e
    # c = U[1*2+0]=U[2], d = U[3]
    # e = V[0]
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[1*2+0][1] = 1  # C21 += M2
    W[1*2+1][1] = -1  # C22 -= M2
    
    # r=2: M3 = a(f-h)
    # a = U[0]
    # f = V[1*2+0]=V[2], h = V[3]
    U[0][2] = 1  # a
    V[2][2] = 1  # f
    V[3][2] = -1  # -h
    W[0*2+1][2] = 1  # C12 += M3
    W[1*2+1][2] = 1  # C22 += M3
    
    # r=3: M4 = d(g-e)
    # d = U[3]
    # g = V[1*2+0]=V[2]... wait, g is B[1,0], so V[1*2+0]=V[2]
    # e = V[0]
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1  # -e
    W[0*2+0][3] = -1  # C11 -= M4
    W[1*2+0][3] = 1  # C21 += M4
    
    # r=4: M5 = (a+b)h
    # a = U[0], b = U[0*2+1]=U[1]
    # h = V[3]
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    W[0*2+0][4] = 1  # C11 += M5
    W[0*2+1][4] = 1  # C12 += M5
    
    # r=5: M6 = (c-d)(f+h)
    # c = U[2], d = U[3]
    # f = V[2], h = V[3]
    U[2][5] = 1  # c
    U[3][5] = -1  # -d
    V[2][5] = 1  # f
    V[3][5] = 1  # h
    W[1*2+1][5] = 1  # C22 += M6
    
    # r=6: M7 = (b-c)(e+g)
    # b = U[1], c = U[2]
    # e = V[0], g = V[2]
    U[1][6] = 1  # b
    U[2][6] = -1  # -c
    V[0][6] = 1  # e
    V[2][6] = 1  # g
    W[0*2+0][6] = -1  # C11 -= M7
    
    return U, V, W

def _recursive_block(n, m, p, depth=0, max_depth=3):
    """
    Recursively decompose n×m×p using block splitting.
    Split m into two halves and p into two halves, creating 4 sub-problems.
    Use Strassen for 2×2×2 base case.
    """
    if time.time() > _DEADLINE - 10.0:
        return None
    
    # Base case: schoolbook for small dimensions
    if n * m * p <= 18 or depth >= max_depth:
        return _schoolbook(n, m, p)
    
    # Try to split m and p
    # Find good splits
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # Recursively decompose the 4 sub-problems
    # But we can be smarter: instead of 4 independent sub-problems,
    # we can use Strassen-like combinations.
    
    # For now, use the simple block combination:
    # C = A[:,:m1]@B[:m1,:p1] + A[:,:m1]@B[:m1,p1:] + A[:,m1:]@B[m1:,:p1] + A[:,m1:]@B[m1:,p1:]
    
    # This is just 4 schoolbook sub-problems combined, which gives rank = n*m*p (no improvement)
    # We need Strassen-like algebraic identities to reduce rank.
    
    # Let's try a different approach: if all dimensions are even, use 2x2x2 Strassen blocks
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        # Use Kronecker product of Strassen with smaller blocks
        # For 2a x 2b x 2c, we can use 7 * (a x b x c) decompositions
        # But we need to be careful about the indexing.
        
        # Actually, let's just do simple recursive halving with Strassen at the base
        if n == 2 and m == 2 and p == 2:
            return _strassen_2x2x2()
    
    # Fallback: split into 4 quadrants and combine
    # C[0:m1, 0:p1] = A[0:m1, :] @ B[:, 0:p1]  -- no, this isn't right
    # C = A @ B where A is n×m, B is m×p
    # Split A = [A1 | A2] (n×m1, n×m2) and B = [B1; B2] (m1×p, m2×p)
    # C = A1@B1 + A2@B2
    
    # This gives 2 sub-problems: (n, m1, p) and (n, m2, p)
    res1 = _recursive_block(n, m1, p, depth + 1, max_depth)
    res2 = _recursive_block(n, m2, p, depth + 1, max_depth)
    
    if res1 is not None and res2 is not None:
        U1, V1, W1 = res1
        U2, V2, W2 = res2
        return _combine_decompositions(n, m1, p, m2, p, U1, V1, W1, U2, V2, W2)
    
    # If that fails, try splitting p instead
    res1 = _recursive_block(n, m, p1, depth + 1, max_depth)
    res2 = _recursive_block(n, m, p2, depth + 1, max_depth)
    
    if res1 is not None and res2 is not None:
        U1, V1, W1 = res1
        U2, V2, W2 = res2
        # C = A@B1 + A@B2 where B1 is m×p1, B2 is m×p2
        # U is same for both, V and W differ
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for r in range(rank1):
                    U[i * m + j][r] = U1[i * m + j][r]
        for j in range(m):
            for k in range(p1):
                for r in range(rank1):
                    V[j * p + k][r] = V1[j * p1 + k][r]
        for k in range(p1):
            for i in range(n):
                for r in range(rank1):
                    W[k * n + i][r] = W1[k * n + i][r]
        
        for i in range(n):
            for j in range(m):
                for r in range(rank2):
                    U[i * m + j][rank1 + r] = U2[i * m + j][r]
        for j in range(m):
            for k in range(p1, p):
                k2 = k - p1
                for r in range(rank2):
                    V[j * p + k][rank1 + r] = V2[j * p2 + k2][r]
        for k in range(p1, p):
            k2 = k - p1
            for i in range(n):
                for r in range(rank2):
                    W[k * n + i][rank1 + r] = W2[k2 * n + i][r]
        
        return U, V, W
    
    return None

def _try_rank_am(n, m, p, rank, max_iter=500, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization (vectorized)."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                A = (V[j * p:(j + 1) * p, :] * W[:, i:i+1].T)  # shape (p, rank)
                A = A.reshape(p, rank)
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = (U[:, j:j+1].T * W[k * n:(k+1)*n, :].T)  # shape (n, rank)
                B = B.reshape(n, rank)
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = (U[i * m:(i+1)*m, :] * V[:, k:k+1].T)  # shape (m, rank)
                C = C.reshape(m, rank)
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # First, try recursive block decomposition (Strassen-based)
    rec_result = _recursive_block(n, m, p, depth=0, max_depth=4)
    if rec_result is not None:
        U, V, W = rec_result
        rec_rank = len(U[0])
        if _verify(U, V, W, n, m, p, tol=1e-6):
            best_result = (U, V, W)
            best_rank = rec_rank
    
    # Try AM search from lower bound upward
    lower_bound = max(n * m, m * p, n * p)
    start_rank = lower_bound if best_result is None else max(lower_bound, best_rank - 10)
    
    for rank in range(start_rank, best_rank + 1):
        if time.time() > _DEADLINE - 15.0:
            break
        
        if rank >= best_rank:
            break
        
        for seed in range(3):
            if time.time() > _DEADLINE - 10.0:
                break
            
            result = _try_rank_am(n, m, p, rank, max_iter=150, tol=1e-8, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_result = (U, V, W)
                    best_rank = rank
                    break
        
        if best_result is not None and best_rank <= start_rank:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
