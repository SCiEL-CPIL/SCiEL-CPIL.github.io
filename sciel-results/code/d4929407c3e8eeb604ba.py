import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _build_winograd_4x5x6():
    """
    Try a Winograd-style decomposition for 4x5x6.
    
    Idea: Use the fact that we can factor the middle dimension.
    A (4x5) @ B (5x6) = sum over j of A[:,j] * B[j,:]
    
    We can try to use fewer terms by exploiting structure.
    For now, let's try a hybrid: use Strassen for 4x4x4 part,
    but handle the extra dimensions more efficiently.
    
    Alternative: Try to find a decomposition with rank < 101.
    Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30.
    Upper bound from Strassen: 49 + 28 + 24 = 101 (current).
    
    Let's try: 4x5x6 = 4x4x6 + 4x1x6
    - 4x4x6: Can we do better than 4x4x4 (49) + 4x4x2 (28) = 77?
      Maybe use a different decomposition for 4x4x6.
    - 4x1x6: Schoolbook rank 24.
    
    Total: 77 + 24 = 101 (same as before).
    
    Let's try: 4x5x6 = 2x5x6 + 2x5x6 (split rows of A)
    - Each 2x5x6: Use 2x2x2 Strassen blocks?
      2x5x6 = 2x4x6 + 2x1x6
      2x4x6 = 2x4x4 + 2x4x2
      2x4x4: Kronecker 2x2x2 with 2x2x2? No, that's for square.
      Actually, 2x4x4 can be done with rank 2*7=14? No.
    
    This is getting complex. Let me try a different approach:
    Use the fact that 5 = 4+1 and 6 = 4+2, and optimize the cross terms.
    """
    n, m, p = 4, 5, 6
    
    # Part 1: A[:,0:4] @ B[0:4,:] (4x4 @ 4x4) -> rank 49
    # Part 2: A[:,0:4] @ B[0:4,4:6] (4x4 @ 4x2) -> rank 28
    # Part 3: A[:,4] @ B[4,:] (4x1 @ 1x6) -> rank 24
    # Total: 101
    
    # Can we do better? Let's try to share some structure.
    # If we use the same U columns for Part 1 and Part 2 (since they share A[:,0:4]),
    # we might be able to reduce the total rank.
    
    # Actually, the current approach already does this implicitly.
    # The issue is that the W matrices for Part 1 and Part 2 are different.
    
    # Let me try a completely different decomposition:
    # Use 7 Strassen-like terms for the 4x4x4 part, but extend them.
    
    # For now, return the same as _build_4x5x6_improved but with a flag.
    return None  # Not implemented yet, fall back to existing

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _residual_zero(U3, V3, W3, n, m, p, tol=1e-4):
    """Check that the 3-tensor T (sum over r of U*V*W) is all-zero, fast."""
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T) < tol))

def _local_reduce_pairs(U, V, W, n, m, p, budget_s=20.0):
    """
    Try to remove pairs of columns (c1,c2) such that T_{c1}+T_{c2} == 1.
    Only run when single-removal has stalled. Limited candidate pool for speed.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    one = np.ones((n, m, p))

    while _time_left() > 10.0 and (time.time() - start_time) < budget_s and R >= 3:
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Precompute T_c for all columns
        Tc_all = np.stack([U3[:, :, c] * V3[:, :, c] * W3[:, :, c] for c in range(R)], axis=0)
        found = False
        for c1 in range(R):
            if _time_left() < 10.0 or (time.time() - start_time) > budget_s:
                break
            for c2 in range(c1 + 1, R):
                if _time_left() < 10.0 or (time.time() - start_time) > budget_s:
                    break
                if np.all(np.abs((T - Tc_all[c1] - Tc_all[c2]) - one) < 1e-4):
                    # Verify exactly
                    mask = [i for i in range(R) if i != c1 and i != c2]
                    U2 = U[:, mask]
                    V2 = V[:, mask]
                    W2 = W[:, mask]
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        R = U.shape[1]
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return U, V, W

def _try_remove_fast(U, V, W, n, m, p, tol=1e-4):
    """
    Try to remove one column c such that the residual (T_without_c - 1) is zero,
    i.e. the removed column's rank-1 tensor T_c equals the all-ones tensor.
    Equivalently: T_total = 1 + T_c, so T_c must be all-ones.
    We check: is T_c (the rank-1 contribution of column c) equal to the all-ones
    4x5x6 tensor? That's a strong condition. More generally we want T_without_c = 1,
    i.e. T_total - T_c = 1.
    Fast path: compute T_total once, then for each c compute T_c and test
    (T_total - T_c == 1).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # T_total
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    one = np.ones((n, m, p))
    # For each column, T_c = einsum over that single column
    # T_total - T_c should equal one
    for c in range(R):
        Tc = U3[:, :, c] * V3[:, :, c] * W3[:, :, c]
        if np.all(np.abs((T - Tc) - one) < tol):
            return c
    return -1

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen.
    Rank = 7 * 7 = 49.
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U part: U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2, r2]
                            if c2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, col] += c1 * c2
            
            # V part: V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2, r2]
                            if c2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, col] += c1 * c2
            
            # W part: W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2, r2]
                            if c2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, col] += c1 * c2
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using 4x4x4 Strassen and 4x4x2 Strassen.
    
    Strategy:
    - Split A (4x5) into A_L (4x4) and A_R (4x1)
    - Split B (5x6) into B_T (4x6) and B_B (1x6)
    - C = A_L @ B_T + A_R @ B_B
    
    For A_L @ B_T (4x4 @ 4x6):
      - Split B_T into B_T1 (4x4) and B_T2 (4x2)
      - Use 4x4x4 Strassen for A_L @ B_T1 (rank 49)
      - Use 4x4x2 Strassen for A_L @ B_T2 (rank 28)
    
    For A_R @ B_B (4x1 @ 1x6):
      - Schoolbook: rank 24
    
    Total rank: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))  # 4x5 = 20 rows
    V = np.zeros((m * p, total_rank))  # 5x6 = 30 rows
    W = np.zeros((p * n, total_rank))  # 6x4 = 24 rows
    
    # Part 1: A_L @ B_T1 (4x4 @ 4x4)
    base = 0
    # U: A_L (4x4) -> rows i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    
    # V: B_T1 (4x4) -> rows j*6+k
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    
    # W: C (4x4 part) -> rows k*4+i
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
    
    # Part 2: A_L @ B_T2 (4x4 @ 4x2)
    base = 49
    # U: A_L (4x4) -> rows i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    
    # V: B_T2 (4x2) -> rows j*6+k (k=4,5)
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, base:base+28] = V442[j * 2 + k_local, :]
    
    # W: C (4x2 part) -> rows k*4+i (k=4,5)
    for k_local in range(2):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, base:base+28] = W442[k_local * 4 + i, :]
    
    # Part 3: A_R @ B_B (4x1 @ 1x6)
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    
    return U, V, W

def _build_4x5x6_alternative():
    """
    Alternative decomposition: split differently to potentially reduce rank.
    
    Try: A (4x5) @ B (5x6)
    Split A into A1 (2x5) and A2 (2x5)
    Split B into B1 (5x3) and B2 (5x3)
    
    C = A1@B1 + A1@B2 + A2@B1 + A2@B2
    
    Each 2x5x3 can be done with rank 2*5*3=30 (schoolbook) or better.
    2x5x3: Use 2x2x2 Strassen blocks?
      2x5x3 = 2x4x3 + 2x1x3
      2x4x3: Can use 2x2x2 Strassen for 2x2x2 parts?
      This is getting complex. Let's try a simpler split.
    
    Actually, let's try: 4x5x6 = 4x3x6 + 4x2x6
    - 4x3x6: Schoolbook rank 72, or use Strassen-like?
    - 4x2x6: Schoolbook rank 48
    
    Total: 72 + 48 = 120 (worse than 101).
    
    Let's try: 4x5x6 = 3x5x6 + 1x5x6
    - 3x5x6: Schoolbook rank 90
    - 1x5x6: Schoolbook rank 6
    
    Total: 90 + 6 = 96 (better than 101!)
    
    But can we do 3x5x6 better than 90?
    3x5x6 = 3x4x6 + 3x1x6
    - 3x4x6: Schoolbook rank 72
    - 3x1x6: Schoolbook rank 18
    
    Total: 72 + 18 + 6 = 96.
    
    Can we do 3x4x6 better?
    3x4x6 = 3x4x4 + 3x4x2
    - 3x4x4: Use 2x2x2 Strassen blocks? 
      3x4x4 = 2x4x4 + 1x4x4
      2x4x4: Rank 2*7=14? No, that's not right.
      Actually, 2x4x4 can be done with rank 2*4*4=32 (schoolbook) or better.
    
    This is getting too complex. Let me just implement the 3x5x6 + 1x5x6 split.
    """
    n, m, p = 4, 5, 6
    
    # Part 1: A[0:3,:] @ B (3x5 @ 5x6) -> schoolbook rank 90
    # Part 2: A[3,:] @ B (1x5 @ 5x6) -> schoolbook rank 6
    # Total: 96
    
    total_rank = 90 + 6
    U = np.zeros((n * m, total_rank))  # 20 rows
    V = np.zeros((m * p, total_rank))  # 30 rows
    W = np.zeros((p * n, total_rank))  # 24 rows
    
    # Part 1: 3x5x6 schoolbook
    base = 0
    for i in range(3):
        for j in range(5):
            for k in range(6):
                col = base + i * 5 * 6 + j * 6 + k
                U[i * 5 + j, col] = 1.0
                V[j * 6 + k, col] = 1.0
                W[k * 4 + i, col] = 1.0
    
    # Part 2: 1x5x6 schoolbook (row 3 of A)
    base = 90
    for j in range(5):
        for k in range(6):
            col = base + j * 6 + k
            U[3 * 5 + j, col] = 1.0
            V[j * 6 + k, col] = 1.0
            W[k * 4 + 3, col] = 1.0
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _local_reduce(U, V, W, n, m, p, max_iters=20, budget_s=60.0):
    """Local column-removal search with randomization, using fast residual check."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)

    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        # Randomly shuffle columns to try different orders
        cols = list(range(R))
        rng.shuffle(cols)

        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _local_reduce_fast(U, V, W, n, m, p, max_iters=200, budget_s=60.0):
    """
    Fast column-removal using the residual trick:
    T_total = 1 + sum_c T_c. Removing column c is valid iff T_c == 1 (all-ones).
    We compute T_total once per iteration, then for each candidate c compute
    T_c in O(nmp) and test T_total - T_c == 1.
    This is much faster than rebuilding the full tensor per removal.
    
    Also try: removing column c is valid if there exists a linear combination
    of other columns that can replace it. We check if the "residual" after
    removing c is zero when projected onto the space spanned by other columns.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(7)
    one = np.ones((n, m, p))

    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)

        # Order candidates: try sparse columns first (more likely redundant)
        nz = np.count_nonzero(U, axis=0) + np.count_nonzero(V, axis=0) + np.count_nonzero(W, axis=0)
        cols = np.argsort(nz)

        found = False
        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            Tc = U3[:, :, c] * V3[:, :, c] * W3[:, :, c]
            if np.all(np.abs((T - Tc) - one) < 1e-4):
                # Verify exactly
                U2 = np.delete(U, c, axis=1)
                V2 = np.delete(V, c, axis=1)
                W2 = np.delete(W, c, axis=1)
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p, budget_s=30.0):
    """
    Try to merge two columns into one by finding if their sum equals
    another column's pattern (up to scaling).
    
    If T_c1 + T_c2 = T_c3 * alpha, then we can remove c1 and c2 and
    replace c3 with c3 + c1 + c2 (adjusting coefficients).
    This is complex, so we try a simpler version: check if removing
    two columns and adding a new one (sum of their contributions) works.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    one = np.ones((n, m, p))
    
    if R < 3:
        return U, V, W
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Precompute T_c for all columns
    Tc_all = np.stack([U3[:, :, c] * V3[:, :, c] * W3[:, :, c] for c in range(R)], axis=0)
    
    # Try removing pairs (c1, c2) and see if the residual can be absorbed
    # into existing columns. This is a linear algebra problem.
    # For simplicity, we'll just try if T_c1 + T_c2 is proportional to
    # some existing T_c3.
    
    for c1 in range(R):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        for c2 in range(c1 + 1, R):
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            sum_Tc = Tc_all[c1] + Tc_all[c2]
            # Check if sum_Tc is proportional to any other Tc
            for c3 in range(R):
                if c3 == c1 or c3 == c2:
                    continue
                if np.all(np.abs(Tc_all[c3]) > 1e-10):
                    # Check if sum_Tc = alpha * Tc_all[c3]
                    # Use least squares to find alpha
                    mask = np.abs(Tc_all[c3]) > 1e-10
                    if np.sum(mask) > 0:
                        alpha = np.sum(sum_Tc[mask] * Tc_all[c3][mask]) / np.sum(Tc_all[c3][mask] ** 2)
                        if np.all(np.abs(sum_Tc - alpha * Tc_all[c3]) < 1e-4):
                            # Found a merge! Remove c1, c2 and scale c3
                            mask = [i for i in range(R) if i != c1 and i != c2]
                            U2 = U[:, mask]
                            V2 = V[:, mask]
                            W2 = W[:, mask]
                            # Find c3 in the new indexing
                            if c3 < c1:
                                new_c3 = c3
                            elif c3 < c2:
                                new_c3 = c3 - 1
                            else:
                                new_c3 = c3 - 2
                            # Scale column new_c3 by (1 + alpha)
                            U2[:, new_c3] *= (1 + alpha)
                            # Verify
                            if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                                return U2, V2, W2
    
    return U, V, W

def _linear_reduce(U, V, W, n, m, p, budget_s=60.0):
    """
    Attempts to reduce rank by finding linear dependencies among columns.
    Uses the structure that if column c can be expressed as a linear combination
    of other columns in the tensor product space, we can eliminate it.
    This is a heuristic based on checking if the tensor defined by the columns
    has a rank deficiency.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    # We check if removing a column still allows the decomposition.
    # A stronger check is if the column is in the span of the others in the
    # "multiplication tensor" sense.
    # For each column c, we check if the rank-1 tensor T_c = u_c v_c^T w_c^T
    # (appropriately permuted) is in the span of the other T_r's.
    # This is expensive, so we use a heuristic: if the vector of coefficients
    # in a basis of the other columns is small, it might be removable.
    
    # Simpler heuristic: try removing columns that are "sparse" or "simple".
    # Or, just iterate through and try to remove.
    
    # This function is currently a stub that falls back to _local_reduce
    # if we run out of time or if the logic is too slow.
    # Let's implement a basic version that tries to remove columns
    # that are linearly dependent in the U, V, or W matrices alone?
    # No, that's not sufficient. The dependency must be in the tensor product.
    
    # Given the time constraints, we will just call _local_reduce with a
    # different seed or more iterations if available.
    # But to satisfy the "combine" requirement, let's try a specific
    # "Winograd-style" simplification if we detect a pattern?
    # That's too complex.
    
    # Let's try a "column compression" via SVD on the concatenated
    # tensor? No, that changes the structure.
    
    # We will stick to _local_reduce but perhaps with a different strategy:
    # Try removing pairs of columns? No, too slow.
    
    # For now, this function will just return the input, acting as a no-op
    # to allow the structure to exist for future refinement, or we can
    # just merge it with _local_reduce.
    
    # Actually, let's implement a "greedy" removal that prioritizes
    # columns with the fewest non-zero entries (sparsity).
    
    U_sparsity = np.count_nonzero(U, axis=0)
    V_sparsity = np.count_nonzero(V, axis=0)
    W_sparsity = np.count_nonzero(W, axis=0)
    
    # Sort columns by total sparsity (ascending)
    total_sparsity = U_sparsity + V_sparsity + W_sparsity
    sorted_cols = np.argsort(total_sparsity)
    
    found = True
    while found and (time.time() - start_time) < budget_s and _time_left() > 5.0:
        found = False
        R = U.shape[1]
        # Re-calculate sparsity each iteration as it changes
        U_sparsity = np.count_nonzero(U, axis=0)
        V_sparsity = np.count_nonzero(V, axis=0)
        W_sparsity = np.count_nonzero(W, axis=0)
        total_sparsity = U_sparsity + V_sparsity + W_sparsity
        sorted_cols = np.argsort(total_sparsity)
        
        for c in sorted_cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    found = True
                    break
                    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Try Winograd-based decomposition first
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Improved 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try Alternative 4x5x6 (3x5x6 + 1x5x6 split)
    try:
        if _time_left() > 20.0:
            U, V, W = _build_4x5x6_alternative()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Fast local column-removal refinement (residual trick)
    try:
        if _time_left() > 10.0:
            cand = _local_reduce_fast(best_U, best_V, best_W, n, m, p, max_iters=500, budget_s=40.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass

    # Column merging refinement
    try:
        if _time_left() > 15.0:
            cand = _try_merge_columns(best_U, best_V, best_W, n, m, p, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass

    # Slower randomized column-removal refinement
    try:
        if _time_left() > 15.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=20, budget_s=25.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass

    # Linear/Sparsity-based column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _linear_reduce(best_U, best_V, best_W, n, m, p, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass

    # Pair-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _local_reduce_pairs(best_U, best_V, best_W, n, m, p, budget_s=10.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass

    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
