"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 10.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _try_round_verify(U, V, W):
    """Round to half-integers and check exactness."""
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    n, m, p = DIMENSIONS
    r = len(Ur[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ur[i * m + j][c] * Vr[j * p + k][c] * Wr[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return None
    return {"U": Ur, "V": Vr, "W": Wr}

def _refine(U, V, W, n, m, p, iters=50):
    """Gauss-Seidel style refinement to drive residual toward zero."""
    r = len(U[0])
    for _ in range(iters):
        # Update W: for each (i,k), solve m linear equations
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass
        # Update U: for each (i,j), solve p linear equations
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass
        # Update V: for each (j,k), solve n linear equations
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
    return U, V, W

def _polish(U, V, W, n, m, p, iters=100):
    """High-precision polish: use tighter tolerance and more iterations."""
    r = len(U[0])
    for _ in range(iters):
        for i in range(n):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)], dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    W[k*n+i] = [float(sol[c]) for c in range(r)]
                except:
                    pass
        for i in range(n):
            for j in range(m):
                A = np.array([[V[j*p+k][c] * W[k*n+i][c] for c in range(r)] for k in range(p)], dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    U[i*m+j] = [float(sol[c]) for c in range(r)]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * W[k*n+i][c] for c in range(r)] for i in range(n)], dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    V[j*p+k] = [float(sol[c]) for c in range(r)]
                except:
                    pass
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    This is the classic Strassen algorithm.
    
    For 2x2*2x2: n=2, m=2, p=2
    U has 4 rows (i*m+j for i,j in 0..1), V has 4 rows, W has 4 rows.
    Each has 7 columns.
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 4 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 4 rows
    
    # M1 = (A11 + A22)(B11 + B22)
    # U for A11 (i=0,j=0): +1, A22 (i=1,j=1): +1
    U[0][0] = 1.0  # A11
    U[3][0] = 1.0  # A22
    V[0][0] = 1.0  # B11
    V[3][0] = 1.0  # B22
    W[0][0] = 1.0  # C11
    W[3][0] = 1.0  # C22
    
    # M2 = (A21 + A22)B11
    U[2][1] = 1.0  # A21
    U[3][1] = 1.0  # A22
    V[0][1] = 1.0  # B11
    W[2][1] = 1.0  # C21
    W[3][1] = -1.0  # C22
    
    # M3 = A11(B12 - B22)
    U[0][2] = 1.0  # A11
    V[1][2] = 1.0  # B12
    V[3][2] = -1.0  # B22
    W[1][2] = 1.0  # C12
    W[3][2] = 1.0  # C22
    
    # M4 = A22(B21 - B11)
    U[3][3] = 1.0  # A22
    V[0][3] = -1.0  # B11
    V[2][3] = 1.0  # B21
    W[0][3] = 1.0  # C11
    W[2][3] = 1.0  # C21
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1.0  # A11
    U[1][4] = 1.0  # A12
    V[3][4] = 1.0  # B22
    W[0][4] = -1.0  # C11
    W[1][4] = 1.0  # C12
    
    # M6 = (A21 - A11)(B11 + B12)
    U[0][5] = -1.0  # A11
    U[2][5] = 1.0  # A21
    V[0][5] = 1.0  # B11
    V[1][5] = 1.0  # B12
    W[3][5] = 1.0  # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1.0  # A12
    U[3][6] = -1.0  # A22
    V[2][6] = 1.0  # B21
    V[3][6] = 1.0  # B22
    W[0][6] = 1.0  # C11
    W[3][6] = 1.0  # C22
    
    return U, V, W

def _build_1x3x6_rank18():
    """
    Build a rank-18 decomposition for 1x3 * 3x6 matrix multiplication.
    This is essentially schoolbook: each output element C[0,k] = sum_j A[0,j] B[j,k]
    requires 3 terms, and there are 6 output elements, so 18 terms.
    But we can potentially do better.
    
    Actually, 1x3 * 3x6 is equivalent to computing 6 dot products of length 3.
    The rank of this tensor is exactly 18 (it's a direct sum of 6 rank-3 tensors).
    
    Wait, let me reconsider. The tensor for 1x3*3x6 has:
    n=1, m=3, p=6
    U has 3 rows, V has 18 rows, W has 6 rows.
    
    For each (j,k) pair where we need the product A[0,j]*B[j,k] to contribute to C[0,k],
    we need a rank-1 term. There are 3*6=18 such pairs, so rank 18 is schoolbook.
    
    Can we do better? The border rank might be lower, but exact rank is likely 18
    for this "diagonal" structure.
    """
    n, m, p = 1, 3, 6
    rank = m * p  # 18
    U = [[0.0] * rank for _ in range(n * m)]  # 3 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 18 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 6 rows
    
    col = 0
    for j in range(m):
        for k in range(p):
            U[j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k][col] = 1.0
            col += 1
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two tensor decompositions into one for the tensor product.
    If T1 has rank r1 and T2 has rank r2, the tensor product T1 ⊗ T2 has rank r1*r2.
    
    For matrix multiplication, if we have:
    T1 for n1 x m1 * m1 x p1 (rank r1)
    T2 for n2 x m2 * m2 x p2 (rank r2)
    
    The product for (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) has rank r1*r2.
    
    But that's not quite what we want. We want to decompose 2x3*3x6.
    
    Let me think differently. We can view 2x3*3x6 as:
    First, multiply the 2x3 matrix by a 3x3 block to get 2x3, then by another 3x3 to get 2x3, then...
    No, that's not right.
    
    Actually, 2x3 * 3x6 can be seen as:
    (2x3) * (3x6) = (2x3) * (3x3 * 3x2) -- no, that's not a valid factorization.
    
    Let me use a different approach. We'll directly construct the decomposition
    using the Strassen structure for the 2x2 part and schoolbook for the rest.
    """
    pass

def _build_hybrid_decomposition():
    """
    Build a hybrid decomposition for 2x3 * 3x6 using Strassen for the 2x2 core.
    
    View the 2x3 matrix as [A_left (2x2) | A_right (2x1)]
    View the 3x6 matrix as [B_top (2x6); B_bottom (1x6)]
    
    C = A * B = [A_left | A_right] * [B_top; B_bottom]
              = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top: 2x2 * 2x6
    For A_right * B_bottom: 2x1 * 1x6
    
    2x2 * 2x6 can be decomposed as 2x2 * (2x3 * 3x2) -- no.
    2x2 * 2x6 = (2x2 * 2x2) * ... no.
    
    Actually, 2x2 * 2x6 is just 6 independent 2x2 * 2x1 multiplications? No.
    2x2 * 2x6: each column of the output is a 2x1 vector = A_left * (column of B_top).
    So it's 6 independent 2x2 * 2x1 multiplications.
    
    2x2 * 2x1: rank 4 (schoolbook) or can we do better?
    C1 = A11*B11 + A12*B21
    C2 = A21*B11 + A22*B21
    This is rank 4. Can we do rank 3? Probably not for exact.
    
    So 2x2 * 2x6 has rank 6*4 = 24 with schoolbook.
    2x1 * 1x6 has rank 2*6 = 12 with schoolbook.
    Total: 24 + 12 = 36. That's just schoolbook.
    
    Let me try a different split.
    
    View 2x3 * 3x6 as:
    A = [[a11, a12, a13], [a21, a22, a23]]
    B is 3x6.
    
    C[0,0] = a11*B[0,0] + a12*B[1,0] + a13*B[2,0]
    C[0,1] = a11*B[0,1] + a12*B[1,1] + a13*B[2,1]
    ...
    C[1,0] = a21*B[0,0] + a22*B[1,0] + a23*B[2,0]
    ...
    
    The key insight: for fixed j, the terms involving A[:,j] and B[j,:] 
    contribute to all C[i,k] via A[i,j]*B[j,k].
    
    Let's group by j:
    For j=0: A[:,0] * B[0,:] contributes to C via outer product of A[:,0] and B[0,:]
    For j=1: A[:,1] * B[1,:] contributes
    For j=2: A[:,2] * B[2,:] contributes
    
    Each such contribution is a 2x6 matrix = (2x1 vector) * (1x6 vector) = rank 1 matrix.
    But in tensor decomposition terms, each (i,k) entry gets A[i,j]*B[j,k].
    
    So for each j, we have a "sub-tensor" that is the outer product of a 2-vector and a 6-vector,
    which in the tensor sense has rank 2*6 = 12? No.
    
    Actually, for fixed j, the tensor T_j[i,k] = A[i,j] * B[j,k] is a rank-1 matrix (outer product),
    but in the tensor decomposition T = sum_j T_j, each T_j is itself a sum of rank-1 tensors.
    T_j[i,k] = A[i,j] * B[j,k] is already a rank-1 tensor (in the sense that it's a product of 
    a function of (i,j) and a function of (j,k)).
    
    Wait, in the tensor decomposition framework:
    T[i,j,k] = 1 if the indices are consistent.
    For fixed j, the "slice" T[:,j,:] has T[i,j,k] = 1 for all i,k.
    This is a 2x6 all-ones matrix, which has rank 1 as a matrix, but as a tensor...
    
    I'm getting confused. Let me just use the random search approach but with better initialization.
    """
    pass

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

def _strassen_schoolbook_hybrid(n, m, p):
    """
    Build a decomposition using Strassen for 2x2 blocks.
    
    For 2x3 * 3x6:
    Split into 2x2 * 2x2 (Strassen, rank 7) and remaining parts.
    
    A = [[A11, A12, A13], [A21, A22, A23]]
    B = 3x6 matrix.
    
    C = A * B
    
    Let's split B into B1 (first 2 rows) and B2 (last 1 row):
    B = [B1; B2] where B1 is 2x6 and B2 is 1x6.
    
    C = [A11, A12, A13] * [B1; B2]
      = [A11, A12] * B1 + A13 * B2
      = [A21, A22] * B1 + A23 * B2
    
    [A11, A12] * B1 is 2x2 * 2x6.
    A13 * B2 is 2x1 * 1x6.
    
    For 2x2 * 2x6, we can use Strassen on the 2x2 part of B1.
    B1 = [B1_left (2x2), B1_right (2x4)]
    
    [A11, A12] * B1 = [A11, A12] * [B1_left | B1_right]
                     = ([A11, A12] * B1_left) | ([A11, A12] * B1_right)
    
    [A11, A12] * B1_left is 2x2 * 2x2, rank 7 (Strassen).
    [A11, A12] * B1_right is 2x2 * 2x4, rank 14 (schoolbook) or can we do better?
    
    2x2 * 2x4: each column is 2x2 * 2x1, rank 4 each, total 16? No, 4 columns * 4 = 16.
    But we can share: 2x2 * 2x4 has rank 8? Let me think.
    
    Actually, 2x2 * 2x4 is equivalent to 4 independent 2x2 * 2x1 multiplications,
    each of rank 4, so total rank 16. But that's worse than schoolbook (16).
    Wait, schoolbook for 2x2*2x4 is 2*2*4 = 16. So no improvement.
    
    Let me try a different approach. Use the full Strassen structure.
    
    Actually, let me just try rank 24 and see if random search can find it.
    """
    return None

def _try_rank(rank, n, m, p, rng, max_attempts=50):
    """Try to find a valid decomposition at given rank via random init + refinement."""
    for attempt in range(max_attempts):
        if _remaining() < 5.0:
            return None
        # Choose initialization strategy based on attempt number
        if attempt % 5 == 0:
            # Sparse half-integer init
            U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 1:
            # Dense small-magnitude init
            U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 2:
            # Structured: some rows share patterns
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        U[i * m + j][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        V[j * p + k][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
            for k in range(p):
                for i in range(n):
                    for c in range(rank):
                        W[k * n + i][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
        elif attempt % 5 == 3:
            # Binary +/- 1 init
            U = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(p * n)]
        else:
            # Gaussian init
            U = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)]
        
        U, V, W = _refine(U, V, W, n, m, p, iters=40)
        res = _residual(U, V, W, n, m, p)
        if res < 1e-7:
            # Try polish
            U, V, W = _polish(U, V, W, n, m, p, iters=50)
            res = _residual(U, V, W, n, m, p)
            if res < 1e-8:
                rounded = _try_round_verify(U, V, W)
                if rounded is not None:
                    return rounded
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Use multiple seeds for diversity
    seeds = [42, 12345, 99999, 7, 31415, 27182, 1000003, 55555, 88888, 123456]
    
    # Try ranks from just below schoolbook down to 18
    for rank in range(best_rank - 1, 17, -1):
        if _remaining() < 15.0:
            break
        found = False
        for seed in seeds:
            if _remaining() < 10.0:
                break
            if found:
                break
            rng = random.Random(seed)
            result = _try_rank(rank, n, m, p, rng, max_attempts=25)
            if result is not None:
                best = result
                best_rank = rank
                found = True
                break
    return best

if __name__ == "__main__":
    print(construct())
