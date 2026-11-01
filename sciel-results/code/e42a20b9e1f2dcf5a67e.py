import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
    U: (4,7) for A indices (i,j)
    V: (4,7) for B indices (j,k)
    W: (4,7) for C indices (k,i)
    """
    # Standard Strassen:
    # C = A*B where A,B are 2x2
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A12-A22)(B11+B12)
    # M7 = (A21-A11)(B12+B22)
    #
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M3+M6
    #
    # A indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11-A22)(B21+B22)
    U[0,0] = 1;  U[3,0] = -1
    V[2,0] = 1;  V[3,0] = 1
    W[0,0] = 1;  W[3,0] = 1
    
    # M2 = (A21+A22)B11
    U[2,1] = 1;  U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A11(B11-B22)
    U[0,2] = 1
    V[0,2] = 1;  V[3,2] = -1
    W[1,2] = 1;  W[3,2] = 1
    
    # M4 = A22(B21-B11)
    U[3,3] = 1
    V[2,3] = 1;  V[0,3] = -1
    W[0,3] = 1;  W[2,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1;  U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1;  W[1,4] = 1
    
    # M6 = (A12-A22)(B11+B12)
    U[1,5] = 1;  U[3,5] = -1
    V[0,5] = 1;  V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A21-A11)(B12+B22)
    U[2,6] = 1;  U[0,6] = -1
    V[1,6] = 1;  V[3,6] = 1
    W[0,6] = 1;  W[2,6] = 1
    
    return U, V, W

def _combine_to_2x3x10():
    """
    Build a rank-55 decomposition for 2x3 x 3x10 -> 2x10.
    
    Strategy: Split the 3x3 middle into two 2x2 blocks sharing one column.
    Actually, a cleaner approach: use Strassen on 2x2 blocks.
    
    For 2x3 x 3x10:
    - View A as 2x3, B as 3x10.
    - Split columns of A into col0, col1, col2.
    - Split rows of B into row0, row1, row2.
    - Split columns of C (output 2x10) into 10 individual columns.
    
    We can use the following block structure:
    - For each pair of adjacent columns in B (and corresponding rows), use Strassen.
    - But that's not quite right either.
    
    Better approach: 
    Rank 60 is schoolbook. To reduce, we need to exploit the small n=2.
    
    Known result: rank(2xm x mxp) can be reduced using Strassen-like methods.
    For 2x3 x 3x10, one approach is to embed into 4x4 blocks and use Strassen.
    
    Actually, let's use a direct algebraic construction.
    
    The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    
    We can write this as a sum of rank-1 terms. The schoolbook has 60 terms.
    We want fewer.
    
    Key insight: For fixed k, the 2x3 tensor in (i,j) is all ones. 
    The rank of a 2x3 all-ones matrix is 1 (it's rank 1: u*v^T where u=[1,1]^T, v=[1,1,1]).
    So for each k, we can decompose the (i,j) part in rank 1.
    But the coupling through j and the output structure...
    
    Let me think about this differently. The multiplication tensor for 2x3 x 3x10:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    As a tensor in (i,j,k): T[i,j,k] = delta_{ik}... no, T[i,j,k] is the coefficient
    such that C[i,k] = sum_{j,r} A[i,j] B[j,k] * (something).
    
    Actually the standard embedding: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    contribute to (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
    
    We need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    
    This is 60 equations. The schoolbook uses 60 rank-1 terms.
    
    To reduce rank, we can try to make some U, V, W entries cancel or combine.
    
    Let's try a structured approach:
    - Use Strassen on the 2x2 part (i in {0,1}, j in {0,1}) for each k.
    - Handle j=2 separately.
    
    For j in {0,1}: We have a 2x2x10 sub-problem (i in {0,1}, j in {0,1}, k in {0,...,9}).
    This can be decomposed using Strassen: for each k, the 2x2 block in (i,j) is all ones.
    Using Strassen, a 2x2x2 multiplication has rank 7 instead of 8.
    
    For 2x2x10: we can use Strassen on the 2x2 part, giving rank 7*10 = 70? No, that's worse.
    
    Wait, Strassen reduces 2x2x2 from 8 to 7. For 2x2xp, the schoolbook is 4p, and Strassen gives 7p? No.
    
    Actually for 2x2xp, the standard rank is 4p. Strassen's method for 2x2x2 gives rank 7.
    For 2x2xp, we can't simply multiply by p because the k dimension couples.
    
    Let me reconsider. The tensor is T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    
    One approach: Use the fact that n=2 is small.
    
    For rank r, we have U (6 x r), V (30 x r), W (20 x r).
    
    Let's try a rank-55 approach based on known results.
    
    Actually, let me try a numerical search with a better initialization.
    """
    pass

def _numerical_search(n, m, p, deadline):
    """
    Search for a low-rank decomposition using least squares with structured initialization.
    """
    num_U = n * m  # 6
    num_V = m * p  # 30
    num_W = p * n  # 20
    num_eq = n * m * p  # 60
    
    # Precompute the index triples
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    
    def residual(x, r):
        U = x[:num_U*r].reshape(num_U, r)
        V = x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
        W = x[num_U*r+num_V*r:].reshape(num_W, r)
        res = np.zeros(num_eq)
        for idx, (iu, iv, iw) in enumerate(indices):
            res[idx] = np.sum(U[iu] * V[iv] * W[iw]) - 1.0
        return res
    
    # Try ranks from 50 to 60
    for r in range(50, 61):
        if time.time() > deadline - 5:
            break
        for attempt in range(3):
            if time.time() > deadline - 3:
                break
            np.random.seed(42 + r * 10 + attempt)
            x0 = np.random.randn((num_U + num_V + num_W) * r) * 0.5
            try:
                sol = scipy_optimize_least_squares(residual, x0, args=(r,), method='lm', max_nfev=500)
                if np.max(np.abs(sol.fun)) < 1e-8:
                    U = sol.x[:num_U*r].reshape(num_U, r)
                    V = sol.x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
                    W = sol.x[num_U*r+num_V*r:].reshape(num_W, r)
                    return U.tolist(), V.tolist(), W.tolist()
            except Exception:
                pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 60)
    U, V, W = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # First, try a structured Strassen-based approach
    # Then, try numerical search
    
    # For now, try numerical search with structured initialization
    result = _numerical_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}
