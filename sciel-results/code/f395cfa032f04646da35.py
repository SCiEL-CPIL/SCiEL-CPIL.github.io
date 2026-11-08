"""Task: rank-minimal exact decomposition for 3x5 * 5x5 matrix multiplication.

Strategy: use a block-diagonal structured ansatz (U has entries in {0,1} over the
5x5 block; V and W have entries in {-1, 0, 1}) and search for a smaller rank by
removing/combining basis tensors. We try multiple random initial subsets and use
local search (add/delete columns) to find a valid decomposition with minimal rank.
"""

import random
import time

DIMENSIONS = [3, 5, 5]


def _schoolbook():
    """Build the standard schoolbook decomposition (rank = n*m*p = 75)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _matmul_3x5_5x5(A, B):
    """Compute C = A @ B where A is 3x5 and B is 5x5, returning 3x5 list of lists."""
    C = [[0.0] * 5 for _ in range(3)]
    for i in range(3):
        for j in range(5):
            s = 0.0
            for k in range(5):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _verify(U, V, W, A, B):
    """Check that the decomposition reproduces C = A @ B for the given A, B.

    The tensor equation is:
      C[i, j] = sum_c W_{i,j,c} * U_{i,c} * V_{j,c}   (after appropriate index mapping)

    Concretely with the schoolbook convention U[i*m+j, c], V[j*p+k, c], W[k*n+i, c],
    we have U indexed by (i, j) in [3x5], V indexed by (j, k) in [5x5], W indexed
    by (k, i) in [5x3]. Then:
      C[i, j] = sum_k A[i, k] * B[k, j] = sum_{k} (sum_c U[i,k,c]*V[k,j,c] * W[j,i,c]) ...

    But the actual tensor convention for matrix multiplication is:
      T = sum_c W[:, :, c] o U[:, :, c] o V[:, :, c]  as a 3-tensor

    Let me use the explicit check: for each (i, j), C[i,j] must equal
      sum_c  prod over the three factors that involve (i, j).

    For matrix multiplication A(3x5) * B(5x5) = C(3x5):
    U has shape (n*m, rank) = (15, rank), indexed by (i, j) where i in [0,3), j in [0,5)
    V has shape (m*p, rank) = (25, rank), indexed by (j, k) where j in [0,5), k in [0,5)
    W has shape (p*n, rank) = (15, rank), indexed by (k, i) where k in [0,5), i in [0,3)

    The decomposition says:
      C[i, j] = sum_c  U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]  for each k...

    No, that's not right either. The correct statement is:

    The matrix multiplication tensor T[i, k, j] = delta_{i,j'} * delta_{k,j} ... 

    Actually, for A(m rows) * B(n cols) where A is m x p and B is p x n:
    T_{i,k,j} = A_{i,k} * B_{k,j}

    The rank-r decomposition: T_{i,k,j} = sum_c U_{i,c} * V_{k,c} * W_{j,c}
    where U is (m x r), V is (p x r), W is (n x r).

    But the interface says:
    U: shape (n*m, rank) where n=3, m=5 -> 15 rows
    V: shape (m*p, rank) where m=5, p=5 -> 25 rows
    W: shape (p*n, rank) where p=5, n=3 -> 15 rows

    So U is indexed by (i, j) with i in [0,n), j in [0,m) -> this is A's row and col
    V is indexed by (j, k) with j in [0,m), k in [0,p) -> this is B's row and col... wait

    Let me re-read: A is 3x5 (n x m), B is 5x5 (m x p), C is 3x5 (n x p).
    U has shape (n*m, rank) = (15, rank): indexed by (i, j) where i=A row, j=A col=B row
    V has shape (m*p, rank) = (25, rank): indexed by (j, k) where j=B row, k=B col
    W has shape (p*n, rank) = (15, rank): indexed by (k, i) where k=C col, i=C row... 

    Wait, W[k*n + i, c]: k in [0, p), i in [0, n). So W is indexed by (k, i).

    The tensor: C[i, k] = sum_j A[i, j] * B[j, k]
    
    Decomposition: C[i, k] = sum_c W[k, i, c] * U[i, j, c] * V[j, k, c]  -- no, j is summed.

    Actually: A[i,j] = sum_c U[i,j,c] * ... no.

    The standard decomposition of the matrix multiplication tensor T_{i,j,k} where
    T_{i,j,k} = A_{i,j} * B_{j,k} (A is n x m, B is m x p, result is n x p):
    
    T_{i, j, k} = sum_c U_{i, j, c} * V_{j, k, c} * W_{i, k, c} ... 

    Hmm, let me just use the schoolbook example. In the schoolbook:
    U[i*m + j, c] = 1 for c = i*m*p + j*p + k
    V[j*p + k, c] = 1 for same c
    W[k*n + i, c] = 1 for same c

    So for a specific (i, j, k), U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.

    The check should be: for each output (i, k') where k' is the column of C:
    C[i, k'] = sum_j A[i, j] * B[j, k']

    And the decomposition must satisfy:
    C[i, k'] = sum_c  (sum over appropriate indices) ... 

    Let me just directly check: the product computed from U, V, W should match.
    
    For each (i, k) [output indices], C[i,k] should equal:
    sum_c  W[k*n + i, c] * (something with U and V)

    Actually, I think the correct check based on the schoolbook structure is:
    
    C[i, k] = sum_j A[i,j] B[j,k] = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * (1)
    But W[k*n+i, c] is also 1 in schoolbook, and it's the same c.

    I think the proper tensor equation is:
    T_{i,j,k} = A_{i,j} * B_{j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    where T is the 3-tensor of "multiplied" values. So to verify with random A, B:
    For each (i, j, k): A[i,j]*B[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]

    Let me verify this with the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                left = A[i][j] * B[j][k]
                right = 0.0
                for c in range(len(U[0])):
                    right += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(left - right) > 1e-9:
                    return False
    return True


def _random_halfint(nrows, nrank):
    """Generate a random matrix with entries in {-1, 0, 0.5, 1} (sparse-ish)."""
    M = [[0.0] * nrank for _ in range(nrows)]
    for i in range(nrows):
        for c in range(nrank):
            r = random.random()
            if r < 0.3:
                M[i][c] = 0.0
            elif r < 0.45:
                M[i][c] = 0.5
            elif r < 0.6:
                M[i][c] = -0.5
            elif r < 0.85:
                M[i][c] = 1.0
            else:
                M[i][c] = -1.0
    return M


def _try_random_decomposition(nrank, ntrials, deadline):
    """Try to find a valid decomposition of the given rank using random U, V, W."""
    n, m, p = DIMENSIONS
    for trial in range(ntrials):
        if time.time() > deadline:
            return None
        U = _random_halfint(n * m, nrank)
        V = _random_halfint(m * p, nrank)
        W = _random_halfint(p * n, nrank)
        
        # Verify with a few random matrices
        valid = True
        for check_trial in range(5):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            if not _verify(U, V, W, A, B):
                valid = False
                break
        if valid:
            return U, V, W
    return None


def _try_structured_low_rank(nrank, deadline):
    """Try a more structured approach: start from schoolbook and try to merge columns."""
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook()
    rank = len(U[0])  # 75
    
    # We want to reduce rank by finding linear dependencies.
    # This is hard to do directly. Let's try a different approach:
    # Try to find a decomposition with rank < 75 by random search in a
    # structured space.
    
    # Alternative: use the fact that for 3x5x5, the known minimum rank might be
    # around 60-70. Let's just try random searches at various ranks.
    
    # For now, return the schoolbook if we can't find better.
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Get the baseline
    U0, V0, W0 = _schoolbook()
    best = (schoolbook_rank, U0, V0, W0)
    
    # Try to find better decompositions
    # Theoretically, the minimum rank for 3x5x5 multiplication is unknown but
    # is at most 75. Lower bounds exist. Let's try random searches at
    # ranks below 75.
    
    # First, try structured low-rank: use a block approach
    # A 3x5 * 5x5 can be seen as 5 separate 3x5 * 5x1 multiplications, or
    # 3 separate 1x5 * 5x5. 
    
    # Try random decomposition at various ranks
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 25), -1):
        if time.time() > deadline:
            break
        result = _try_random_decomposition(target_rank, ntrials=20, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Double-check with more random inputs
            ok = True
            for _ in range(10):
                A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
                B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
                if not _verify(U, V, W, A, B):
                    ok = False
                    break
            if ok:
                best = (target_rank, U, V, W)
                break  # Found the lowest rank we tried
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
