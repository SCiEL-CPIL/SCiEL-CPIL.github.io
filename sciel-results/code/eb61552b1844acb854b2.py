import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36

    # Block-repetition strategy: find rank-r for 2x3x2, expand to 2x3x6 (rank 3r)
    n_sub, m_sub, p_sub = 2, 3, 2
    repeats = p // p_sub  # 3

    # Lower bound for sub-problem: max(n_sub*m_sub, m_sub*p_sub, p_sub*n_sub) = max(6,6,4)=6
    # But rank-3 might work for 2x3x2? No, lower bound is 6. Try 6,7,8,...
    # Full rank lower bound is 18, so sub-rank lower bound is 6.
    min_sub_rank = max(n_sub * m_sub, m_sub * p_sub, p_sub * n_sub)  # 6

    for sub_rank in range(min_sub_rank, best_rank // repeats + 1):
        if time.time() > deadline:
            break

        full_rank = sub_rank * repeats
        if full_rank >= best_rank:
            break

        # Budget per sub_rank: more attempts for smaller ranks (harder to hit but bigger gain)
        max_attempts = 500 if sub_rank <= 7 else 200
        found = _search_sub_decomposition(n_sub, m_sub, p_sub, sub_rank, max_attempts, deadline)

        if found is not None:
            U_sub, V_sub, W_sub = found
            result = _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub)
            if result is not None and full_rank < best_rank:
                best_result = result
                best_rank = full_rank
                # If we hit the theoretical lower bound, stop
                if best_rank <= max(n * m, m * p, p * n):
                    break

    return best_result


def _schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _search_sub_decomposition(n, m, p, r, max_attempts, deadline):
    """
    Search for a rank-r decomposition of n×m × m×p matmul tensor.
    Returns (U, V, W) as lists of lists, or None.
    
    Key condition: A_{(i,k), (j,c)} = U_{i,j,c} * V_{j,k,c} must have rank n*p.
    If so, W = A^{-1} (or pseudo-inverse) gives an exact decomposition.
    """
    np_sub = n * p  # 4
    nm = n * m     # 6
    mp = m * p     # 6

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        # Generate random half-integer U and V with small values
        # Use values from {-1, 0, 1} to keep the product matrix small and rank-deficient-likely
        U_rand = np.random.randint(-1, 2, size=(nm, r))
        V_rand = np.random.randint(-1, 2, size=(mp, r))

        # Build A: shape (n*p, m*r)
        # A[(i,k), (j,c)] = U[i*m+j, c] * V[j*p+k, c]
        A = np.zeros((np_sub, m * r))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for c in range(r):
                        A[i * p + k, j * r + c] = U_rand[i * m + j, c] * V_rand[j * p + k, c]

        # Check rank of A
        try:
            rank_A = np.linalg.matrix_rank(A, tol=1e-9)
        except np.linalg.LinAlgError:
            continue

        if rank_A < np_sub:
            continue

        # A has full row rank (np_sub). Solve for W: A @ W = I_{np_sub}
        # W shape: (m*r, n*p)
        # Use least squares / pseudo-inverse
        try:
            W_solve = np.linalg.lstsq(A, np.eye(np_sub), rcond=1e-12)[0]
        except np.linalg.LinAlgError:
            continue

        # W_solve should be exact if A has full rank and we're in exact arithmetic.
        # Check if W_solve entries are close to half-integers.
        W_rounded = np.round(W_solve * 2) / 2.0

        # Verify: A @ W_rounded should equal I
        if not np.allclose(A @ W_rounded, np.eye(np_sub), atol=1e-9):
            continue

        # Also verify the full tensor equation explicitly
        if _verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rounded.tolist(), n, m, p):
            return U_rand.tolist(), V_rand.tolist(), W_rounded.tolist()

    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    r = U_np.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub):
    """
    Expand a 2x3x2 decomposition to 2x3x6 by repeating it p/p_sub times.
    U_sub: (n*m, r_sub), V_sub: (m*p_sub, r_sub), W_sub: (p_sub*n, r_sub)
    """
    r_sub = len(U_sub[0])
    repeats = p // p_sub
    r_total = r_sub * repeats

    U_full = [[0] * r_total for _ in range(n * m)]
    V_full = [[0] * r_total for _ in range(m * p)]
    W_full = [[0] * r_total for _ in range(p * n)]

    for block in range(repeats):
        offset = block * r_sub

        # Copy U (same for all blocks)
        for i in range(n * m):
            for c in range(r_sub):
                U_full[i][offset + c] = U_sub[i][c]

        # Copy V (shift columns by block)
        for j in range(m):
            for k in range(p_sub):
                for c in range(r_sub):
                    V_full[j * p + block * p_sub + k][offset + c] = V_sub[j * p_sub + k][c]

        # Copy W (shift rows by block)
        for i in range(n):
            for k in range(p_sub):
                for c in range(r_sub):
                    W_full[(block * p_sub + k) * n + i][offset + c] = W_sub[k * n + i][c]

    # Final verification
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    return None
# EVOLVE-BLOCK-END
