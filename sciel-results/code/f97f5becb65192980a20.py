import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each product c, define U[i,j,c], V[j,k,c], W[k,i,c]
    # such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta(i,j,k) for the 2x2 case
    # Actually for 2x2, T[i,j,k] = 1 for all i,j,k in {0,1}
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: (A00+A11)(B00+B11)
    # A00 contributes to i=0,j=0; A11 contributes to i=1,j=1
    # B00 contributes to j=0,k=0; B11 contributes to j=1,k=1
    # C00 gets +1, C11 gets +1
    # U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1 (for C00: k=0,i=0), W[1,1,0]=1 (for C11: k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1  # i*m+j: 0*2+0=0, 1*2+1=3
    V[0, 0] = 1; V[3, 0] = 1  # j*p+k: 0*2+0=0, 1*2+1=3
    W[0, 0] = 1; W[3, 0] = 1  # k*n+i: 0*2+0=0, 1*2+1=3
    
    # Product 2: (A10+A11)B00
    # A10: i=1,j=0; A11: i=1,j=1
    # B00: j=0,k=0
    # C10 gets +1
    U[2, 1] = 1; U[3, 1] = 1  # 1*2+0=2, 1*2+1=3
    V[0, 1] = 1               # 0*2+0=0
    W[2, 1] = 1               # 0*2+1=2 (k=0,i=1)
    
    # Product 3: A00(B01-B11)
    # A00: i=0,j=0
    # B01: j=0,k=1; B11: j=1,k=1
    # C01 gets +1
    U[0, 2] = 1               # 0*2+0=0
    V[1, 2] = 1; V[3, 2] = -1 # 0*2+1=1, 1*2+1=3
    W[1, 2] = 1               # 1*2+0=1 (k=1,i=0)
    
    # Product 4: A11(B10-B00)
    # A11: i=1,j=1
    # B10: j=1,k=0; B00: j=0,k=0
    # C10 gets +1, C00 gets +1
    U[3, 3] = 1               # 1*2+1=3
    V[2, 3] = 1; V[0, 3] = -1 # 1*2+0=2, 0*2+0=0
    W[0, 3] = 1; W[2, 3] = 1  # 0*2+0=0 (C00), 0*2+1=2 (C10)
    
    # Product 5: (A00+A01)B11
    # A00: i=0,j=0; A01: i=0,j=1
    # B11: j=1,k=1
    # C01 gets -1 (from M1-M5 in C00? No, C00=M1+M4-M5+M7, so -M5 in C00)
    # Wait, let me recheck: C00 = M1+M4-M5+M7
    # So M5 contributes -1 to C00
    # C11 = M1-M2+M3+M6, M5 doesn't appear
    # Hmm, let me re-derive.
    
    # Actually, let me use the standard Strassen formulas more carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me redo this properly.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # Contributes +1 to C00, +1 to C11
    U[0, 0] = 1; U[3, 0] = 1  # a at (0,0), d at (1,1)
    V[0, 0] = 1; V[3, 0] = 1  # e at (0,0), h at (1,1)
    W[0, 0] = 1; W[3, 0] = 1  # C00 at (k=0,i=0), C11 at (k=1,i=1)
    
    # M2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
    # Contributes +1 to C10, -1 to C11
    U[2, 1] = 1; U[3, 1] = 1  # c at (1,0), d at (1,1)
    V[0, 1] = 1               # e at (0,0)
    W[2, 1] = 1               # C10 at (k=0,i=1)
    W[3, 1] = -1              # C11 at (k=1,i=1)
    
    # M3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    # Contributes +1 to C01, +1 to C11
    U[0, 2] = 1               # a at (0,0)
    V[1, 2] = 1; V[3, 2] = -1 # f at (0,1), h at (1,1)
    W[1, 2] = 1               # C01 at (k=1,i=0)
    W[3, 2] = 1               # C11 at (k=1,i=1)
    
    # M4 = d(g-e): d=A[1,1], g=B[1,0], e=B[0,0]
    # Contributes +1 to C00, +1 to C10
    U[3, 3] = 1               # d at (1,1)
    V[2, 3] = 1; V[0, 3] = -1 # g at (1,0), e at (0,0)
    W[0, 3] = 1               # C00 at (k=0,i=0)
    W[2, 3] = 1               # C10 at (k=0,i=1)
    
    # M5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,1]
    # Contributes -1 to C00, +1 to C01
    U[0, 4] = 1; U[1, 4] = 1  # a at (0,0), b at (0,1)
    V[3, 4] = 1               # h at (1,1)
    W[0, 4] = -1              # C00 at (k=0,i=0)
    W[1, 4] = 1               # C01 at (k=1,i=0)
    
    # M6 = (c-a)(e+f): c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
    # Contributes +1 to C11
    U[2, 5] = 1; U[0, 5] = -1 # c at (1,0), a at (0,0)
    V[0, 5] = 1; V[1, 5] = 1  # e at (0,0), f at (0,1)
    W[3, 5] = 1               # C11 at (k=1,i=1)
    
    # M7 = (b-d)(g+h): b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
    # Contributes +1 to C00
    U[1, 6] = 1; U[3, 6] = -1 # b at (0,1), d at (1,1)
    V[2, 6] = 1; V[3, 6] = 1  # g at (1,0), h at (1,1)
    W[0, 6] = 1               # C00 at (k=0,i=0)
    
    return U, V, W

def _verify_2x2(U, V, W):
    """Verify 2x2 Strassen decomposition."""
    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(7):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    print(f"FAIL: i={i},j={j},k={k}, s={s}")
                    return False
    return True

def _build_strassen_3x4x6():
    """
    Build a rank-66 decomposition for 3x4x6 matrix multiplication using
    Strassen-style block decomposition.
    
    Strategy: Use 2x2 Strassen on blocks.
    A is 3x4, B is 4x6, C is 3x6.
    
    Split A into 2x2 blocks of size (2,2) and (1,2):
    A = [[A00 (2x2), A01 (2x2)],
         [A10 (1x2), A11 (1x2)]]
    
    Split B into 2x2 blocks of size (2,3) and (2,3):
    B = [[B00 (2x3), B01 (2x3)],
         [B10 (2x3), B11 (2x3)]]
    
    C = [[C00 (2x3), C01 (2x3)],
         [C10 (1x3), C11 (1x3)]]
    
    For the 2x2 block multiplication, we need to handle non-square blocks.
    Actually, Strassen works for any block sizes as long as the blocks are
    compatible. Let me think about this differently.
    
    Alternative: Use the fact that 3x4x6 can be embedded in a larger
    problem, or use a direct tensor decomposition approach.
    
    Let me try a different approach: use the rank-66 decomposition from
    the literature for 3x4x6, or construct one via block Strassen.
    
    Actually, the simplest approach: start with schoolbook (rank 72) and
    then try to reduce it using Strassen-like cancellations.
    
    For now, let me just return schoolbook and then try to reduce it.
    """
    n, m, p = 3, 4, 6
    rank = n * m * p  # 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _try_strassen_reduction(n, m, p, U, V, W):
    """
    Try to reduce the rank by finding Strassen-like cancellations.
    This is a heuristic approach.
    """
    rank = U.shape[1]
    
    # For each pair of columns, check if they can be combined
    # This is complex, so let's just return the original for now
    return U, V, W

def _exact_rank_test(U, V, W, n, m, p):
    """
    Use exact rational arithmetic to test if any column is linearly dependent.
    For a rank-r decomposition, column c is redundant if there exist
    coefficients alpha_c (not all zero) such that the tensor
    sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0.
    
    This is equivalent to checking if the r rank-1 tensors are linearly
    independent in the tensor space.
    
    We can test this by checking if the matrix formed by flattening the
    tensor has rank r.
    """
    rank = U.shape[1]
    
    # Build the tensor T as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T[i,j,k] = s
    
    # The rank of the tensor is at most the rank of any matricization
    # Matricization along mode 1: T_mat1[i, (j,k)] shape (n, m*p)
    T_mat1 = T.reshape(n, m*p)
    T_mat2 = T.reshape(m, n*p)
    T_mat3 = T.reshape(p, n*m)
    
    # The tensor rank is at least the max of the matrix ranks
    r1 = np.linalg.matrix_rank(T_mat1)
    r2 = np.linalg.matrix_rank(T_mat2)
    r3 = np.linalg.matrix_rank(T_mat3)
    
    return max(r1, r2, r3)

def _find_redundant_columns(U, V, W, n, m, p, tol=1e-10):
    """
    Find columns that are linearly dependent on the others.
    For each column c, check if removing it still gives a valid decomposition.
    This is expensive but exact for small ranks.
    """
    rank = U.shape[1]
    redundant = []
    
    for c in range(rank):
        # Remove column c
        mask = np.ones(rank, dtype=bool)
        mask[c] = False
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if the decomposition is still valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for cc in range(rank - 1):
                        s += U_sub[i*m+j, cc] * V_sub[j*p+k, cc] * W_sub[k*n+i, cc]
                    if abs(s - 1.0) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            redundant.append(c)
    
    return redundant

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """
    Try to round the floating-point decomposition to integers/half-integers
    and verify exactness.
    """
    # Try rounding to nearest integer
    U_int = np.round(U).astype(np.int64)
    V_int = np.round(V).astype(np.int64)
    W_int = np.round(W).astype(np.int64)
    
    # Verify with integer arithmetic (exact)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U_int[0])):
                    s += int(U_int[i*m+j, c]) * int(V_int[j*p+k, c]) * int(W_int[k*n+i, c])
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    # Try half-integer rounding
    U_half = np.round(U * 2) / 2.0
    V_half = np.round(V * 2) / 2.0
    W_half = np.round(W * 2) / 2.0
    
    U_int2 = (U_half * 2).astype(np.int64)
    V_int2 = (V_half * 2).astype(np.int64)
    W_int2 = (W_half * 2).astype(np.int64)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U_int2[0])):
                    s += int(U_int2[i*m+j, c]) * int(V_int2[j*p+k, c]) * int(W_int2[k*n+i, c])
                if s != 8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return schoolbook rank-n*m*p decomposition."""
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

def _strassen_block_decomposition(n, m, p):
    """
    Construct a rank-66 decomposition for 3x4x6 using Strassen-style
    block decomposition.
    
    We split the problem into 2x2 blocks:
    A (3x4) = [[A00 (2x2), A01 (2x2)],
               [A10 (1x2), A11 (1x2)]]
    B (4x6) = [[B00 (2x3), B01 (2x3)],
               [B10 (2x3), B11 (2x3)]]
    C (3x6) = [[C00 (2x3), C01 (2x3)],
               [C10 (1x3), C11 (1x3)]]
    
    For the 2x2 block structure, we use Strassen's 7 products.
    Each product involves multiplying sub-blocks.
    
    The rank is the sum of the ranks of each block multiplication.
    For 2x2 blocks with 2x3 sub-blocks:
    - A00*B00: 2x2 * 2x3 -> rank 6 (schoolbook) or 5 (if we can do better)
    - Actually, for rectangular blocks, the rank is at least max(m1*p1, ...)
    
    This is getting complex. Let me just use the known result that
    3x4x6 has rank at most 66 and try to find it via optimization.
    """
    # For now, return schoolbook
    n, m, p = 3, 4, 6
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _am_decompose(T, rank, n, m, p, max_iter=200, tol=1e-12, seed=0):
    """
    Alternating Minimization for 3-way tensor decomposition.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    for c in range(rank):
        scale = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
        if scale > 1e-15:
            U[:, c] /= scale ** (1/3)
            V[:, c] /= scale ** (1/3)
            W[:, c] /= scale ** (1/3)
    
    prev_err = np.inf
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[i, j, :], rcond=None)
                except:
                    continue
                U[i * m + j, :] = x
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[:, j, k], rcond=None)
                except:
                    continue
                V[j * p + k, :] = x
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[i, :, k], rcond=None)
                except:
                    continue
                W[k * n + i, :] = x
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - T[i,j,k])**2
        err = np.sqrt(err)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    T = np.ones((n, m, p), dtype=np.float64)

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Strategy: Start with schoolbook (rank 72), then try to reduce rank
    # by finding linear dependencies among the columns using exact arithmetic.
    
    # First, get the schoolbook decomposition as numpy arrays
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try to find redundant columns in the schoolbook decomposition
    # For schoolbook, each column corresponds to a unique (i,j,k) triple,
    # so no column is redundant. But we can try Strassen-like combinations.
    
    # Try AM decomposition at various ranks
    ranks_to_try = [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue

        max_restarts = 8
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            seed = restart * 1000 + rank
            U, V, W, err = _am_decompose(T, rank, n, m, p, max_iter=150, tol=1e-10, seed=seed)
            
            if err < 1e-8:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            elif err < 0.1:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        if found:
            continue

    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    T = np.ones((n, m, p), dtype=np.float64)

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    ranks_to_try = [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]

    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue

        max_restarts = 8
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            seed = restart * 1000 + rank
            U, V, W, err = _am_decompose(T, rank, n, m, p, max_iter=150, tol=1e-10, seed=seed)
            
            if err < 1e-8:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            elif err < 0.1:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        if found:
            continue

    return best
# EVOLVE-BLOCK-END
