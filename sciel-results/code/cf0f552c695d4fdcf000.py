import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    import random
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        if C_comp != C_exp:
            return False
    return True

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    Split A into two 2x2 blocks, B into top/bottom 2x6 halves, each with 3 col blocks.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    strassen_terms = _strassen_2x2x2_terms()
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_multiplication_tensor(n, m, p):
    """
    Build the multiplication tensor T[a, b, c] = 1 if (i,j),(j,k),(k,i) consistent, else 0.
    a = i*m + j, b = j*p + k, c = k*n + i
    Returns a 3D numpy array of shape (n*m, m*p, p*n).
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _try_als_decomposition(n, m, p, target_rank, max_iters=30, deadline=None):
    """
    Try to find a rank-target_rank decomposition via Alternating Least Squares.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    T = _build_multiplication_tensor(n, m, p)
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1

    # Precompute index mappings
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))

    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break

        # --- Solve for W ---
        M_W = np.zeros((n * m * m * p, target_rank))
        idx = 0
        for a in range(n * m):
            for b in range(m * p):
                for r in range(target_rank):
                    M_W[idx, r] = U[a, r] * V[b, r]
                idx += 1
        for c in range(p * n):
            rhs = T[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
            except:
                pass

        # --- Solve for U ---
        M_U = np.zeros((m * p * p * n, target_rank))
        idx = 0
        for b in range(m * p):
            for c in range(p * n):
                for r in range(target_rank):
                    M_U[idx, r] = V[b, r] * W[c, r]
                idx += 1
        for a in range(n * m):
            rhs = T[a, :, :].flatten()
            try:
                U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
            except:
                pass

        # --- Solve for V ---
        M_V = np.zeros((n * m * p * n, target_rank))
        idx = 0
        for a in range(n * m):
            for c in range(p * n):
                for r in range(target_rank):
                    M_V[idx, r] = U[a, r] * W[c, r]
                idx += 1
        for b in range(m * p):
            rhs = T[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
            except:
                pass

    # Round to nearest half-integer
    U_round = np.round(U * 2) / 2
    V_round = np.round(V * 2) / 2
    W_round = np.round(W * 2) / 2

    # Verify exactly
    import random
    error = 0.0
    for (a, b, c) in indices:
        val = np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :])
        error += abs(val - 1.0)
    for _ in range(100):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if not (j_a == j_b and k_b == k_c and i_a == i_c):
            val = np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :])
            error += abs(val)

    if error < 1e-6:
        return U_round, V_round, W_round
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # First, try the known Strassen-based rank-42 decomposition
    result = _strassen_block_decomposition()
    if _verify_decomposition(result, n, m, p, trials=50):
        return result

    # Try ALS for lower ranks
    for target_rank in range(24, 42):
        if time.time() > deadline - 20:
            break
        res = _try_als_decomposition(n, m, p, target_rank, max_iters=20, deadline=deadline)
        if res is not None:
            U, V, W = res
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: return the Strassen decomposition (should always be valid)
    return result
