import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
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
    Build the multiplication tensor T[a, b, c].
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

def _local_search_discrete(U, V, W, T, n, m, p, max_iters=50, deadline=None):
    """
    Perform local search on the half-integer lattice to minimize error.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute indices for non-zero T entries and some zero entries for checking
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
                
    # Sample zero indices
    zero_indices = []
    while len(zero_indices) < 100:
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
            zero_indices.append((a, b, c))
            
    all_check_indices = non_zero_indices + zero_indices
    
    def calc_error():
        err = 0.0
        for a, b, c in all_check_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            target = T[a, b, c]
            err += (val - target) ** 2
        return err

    current_error = calc_error()
    if current_error < 1e-9:
        return U, V, W, True

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        improved = False
        
        # Try modifying U
        for a in range(n * m):
            for r in range(len(U[0])):
                orig = U[a, r]
                for delta in [-0.5, 0.5]:
                    U[a, r] = orig + delta
                    new_err = calc_error()
                    if new_err < current_error - 1e-9:
                        current_error = new_err
                        improved = True
                        break
                    else:
                        U[a, r] = orig
                
        # Try modifying V
        for b in range(m * p):
            for r in range(len(V[0])):
                orig = V[b, r]
                for delta in [-0.5, 0.5]:
                    V[b, r] = orig + delta
                    new_err = calc_error()
                    if new_err < current_error - 1e-9:
                        current_error = new_err
                        improved = True
                        break
                    else:
                        V[b, r] = orig
                        
        # Try modifying W
        for c in range(p * n):
            for r in range(len(W[0])):
                orig = W[c, r]
                for delta in [-0.5, 0.5]:
                    W[c, r] = orig + delta
                    new_err = calc_error()
                    if new_err < current_error - 1e-9:
                        current_error = new_err
                        improved = True
                        break
                    else:
                        W[c, r] = orig
                        
        if not improved:
            break
            
    if current_error < 1e-9:
        return U, V, W, True
    return None, None, None, False

def _try_als_decomposition(n, m, p, target_rank, max_iters=50, deadline=None, seed=42):
    """
    Try to find a rank-target_rank decomposition via Alternating Least Squares.
    """
    T = _build_multiplication_tensor(n, m, p)
    np.random.seed(seed)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1

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
    error = 0.0
    for (a, b, c) in indices:
        val = np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :])
        error += abs(val - 1.0)
    
    # Check zero entries
    for _ in range(50):
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

    # If close, try local search
    if error < 1.0:
        U_ls, V_ls, W_ls, found = _local_search_discrete(U_round, V_round, W_round, T, n, m, p, max_iters=20, deadline=deadline)
        if found:
            return U_ls, V_ls, W_ls
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # First, try the known Strassen-based rank-42 decomposition
    result = _strassen_block_decomposition()
    if _verify_decomposition(result, n, m, p, trials=50):
        best_result = result
        best_rank = 42
    else:
        # Should not happen, but fallback to schoolbook if Strassen fails verification (unlikely)
        best_result = None
        best_rank = 48

    # Try ALS for lower ranks
    # Theoretical lower bound: 24
    # Try from 24 up to best_rank - 1
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 30:
            break
            
        found = False
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            res = _try_als_decomposition(n, m, p, target_rank, max_iters=30, deadline=deadline, seed=seed)
            if res is not None:
                U, V, W = res
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                candidate = {"U": U_list, "V": V_list, "W": W_list}
                if _verify_decomposition(candidate, n, m, p, trials=20):
                    best_result = candidate
                    best_rank = target_rank
                    found = True
                    break
        if found:
            # If we found a lower rank, we can stop or continue to see if we can go lower
            # But usually lower ranks are harder. Let's continue to try lower ranks if time permits.
            # Actually, the loop goes from 24 up. So if we find 30, we stop trying 31..41?
            # No, the loop is `for target_rank in range(24, best_rank)`.
            # If best_rank was 42, and we find 30, we update best_rank to 30.
            # The range is fixed at the start of the loop? No, `range(24, best_rank)` is evaluated once.
            # So it will try 24..41.
            # If we find 30, we should break out of the inner seed loop, but continue the outer loop?
            # No, if we found 30, we want to see if we can find 29, 28...
            # So we should NOT break the outer loop.
            # But we should update best_rank so that if we find 29, we update again.
            # The current logic updates best_rank.
            # However, if we find 30, we continue to try 29, 28... in the next iterations of the outer loop?
            # No, the outer loop iterates `target_rank` from 24 to 41.
            # If we find a solution for 30, we set `best_rank = 30`.
            # The loop continues to 31? No, `range(24, 42)` is fixed.
            # So it will try 31, 32... which are worse.
            # We should break if we found a solution and we are moving to higher ranks?
            # No, we are moving from low to high.
            # If we find 30, we want to keep looking for 29, 28...
            # So we should NOT break.
            # But we should skip ranks higher than the best found?
            # The loop goes 24, 25, ..., 41.
            # If we find 30, we have already tried 24-29.
            # So we are done. We can break the outer loop.
            break

    if best_result is None:
        best_result = _strassen_block_decomposition()

    return best_result
