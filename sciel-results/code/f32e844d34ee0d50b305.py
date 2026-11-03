import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _strassen_2x2():
    """Rank-7 decomposition for 2x2 x 2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1

    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W


def _build_4x4x5_from_strassen():
    """Build 4x4x5 decomposition: Strassen 4x4x4 core + boundary terms."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 16
    rank_total = rank_core + rank_boundary

    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    for c in range(rank_core):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] = U44[i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_iter=200, tol=1e-9):
    """Attempt to reduce rank by removing provably-redundant rank-1 terms."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])

    for _ in range(max_iter):
        removed = False
        for c in range(rank - 1, -1, -1):
            U_test = [row[:c] + row[c + 1:] for row in U]
            V_test = [row[:c] + row[c + 1:] for row in V]
            W_test = [row[:c] + row[c + 1:] for row in W]
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break

    if rank < len(U[0]) or rank < len(V[0]) or rank < len(W[0]):
        return U, V, W, rank
    return None


def _try_merge_terms(U, V, W, n, m, p, tol=1e-9):
    """Try to find pairs of terms that can be merged (sum/diff gives valid decomposition).
    This is a limited search for algebraic simplifications."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    best_U, best_V, best_W, best_rank = U, V, W, rank

    # Try removing each term and checking
    for c in range(rank - 1, -1, -1):
        U_test = [row[:c] + row[c + 1:] for row in U]
        V_test = [row[:c] + row[c + 1:] for row in V]
        W_test = [row[:c] + row[c + 1:] for row in W]
        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
            U, V, W = U_test, V_test, W_test
            rank -= 1
            best_U, best_V, best_W, best_rank = U, V, W, rank

    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Build structured Strassen-based 4x4x5 decomposition
    if time.time() < deadline:
        U_new, V_new, W_new = _build_4x4x5_from_strassen()
        if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
            rank_total = len(U_new[0])
            best_rank = rank_total
            best_result = {"U": U_new, "V": V_new, "W": W_new}

            # Attempt rank reduction
            if time.time() < deadline:
                reduced = _try_reduce_rank(U_new, V_new, W_new, n, m, p, max_iter=300)
                if reduced is not None:
                    U_r, V_r, W_r, rank_r = reduced
                    if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                        best_rank = rank_r
                        best_result = {"U": U_r, "V": V_r, "W": W_r}

    # Try to find lower-rank solutions via structured search
    if time.time() < deadline and best_rank > 30:
        # Strategy: Try various rank values and search for valid decompositions
        # using a structured parameterization inspired by Strassen patterns.
        
        # Approach 1: Try to find a decomposition where the boundary terms
        # can be absorbed into the Strassen core terms.
        # The key idea: for the 5th column, C[:,4] = A @ B[:,4]
        # We need sum_c U[i,j,c] * V[j,4,c] * W[4,i,c] = 1 for all i,j
        # while also maintaining sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for k=0..3
        
        # Let's try a direct linear algebra approach:
        # Fix U and V from the Strassen core, solve for W.
        if time.time() < deadline:
            U_core, V_core, W_core = _build_4x4x4_strassen()
            rank_core = 28
            
            # For the 5th column, we need additional rank-1 terms
            # Try different numbers of additional terms
            for extra_rank in range(0, 20):
                if time.time() > deadline:
                    break
                
                total_rank = rank_core + extra_rank
                U_try = [row[:] for row in U_core] + [[0] * rank_core for _ in range(16)]
                V_try = [row[:] for row in V_core] + [[0] * rank_core for _ in range(20)]
                W_try = [row[:] for row in W_core] + [[0] * rank_core for _ in range(20)]
                
                # Extend to full size
                U_full = [[0] * total_rank for _ in range(16)]
                V_full = [[0] * total_rank for _ in range(20)]
                W_full = [[0] * total_rank for _ in range(20)]
                
                for c in range(rank_core):
                    for i in range(16):
                        U_full[i][c] = U_core[i][c]
                    for j in range(16):
                        V_full[j][c] = V_core[j][c]
                    for k in range(16):
                        W_full[k][c] = W_core[k][c]
                
                # For the extra terms, we need to fill in the 5th column
                # The constraint is: for each (i,j), sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
                # And for k=0..3: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                
                # This is complex; skip for now and try simpler approaches
                pass

    # Try random search with small entry values
    if time.time() < deadline and best_rank > 35:
        import random
        rng = random.Random(42)
        
        # Try ranks from 25 to best_rank-1
        for target_rank in range(25, min(best_rank, 45)):
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            attempts = min(50, int((deadline - time.time()) / 0.3))
            for _ in range(attempts):
                if time.time() > deadline:
                    break

                U = [[0] * target_rank for _ in range(16)]
                V = [[0] * target_rank for _ in range(20)]
                W = [[0] * target_rank for _ in range(20)]

                # Use sparse random entries with values from {-1, 0, 1}
                density = max(0.1, 0.5 - target_rank / 100)
                for c in range(target_rank):
                    for i in range(16):
                        if rng.random() < density:
                            U[i][c] = rng.choice([-1, 1])
                    for i in range(20):
                        if rng.random() < density:
                            V[i][c] = rng.choice([-1, 1])
                            W[i][c] = rng.choice([-1, 1])

                if _verify_decomposition_fast(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break

    # Final: try to reduce the best result further
    if time.time() < deadline:
        U_best, V_best, W_best = best_result["U"], best_result["V"], best_result["W"]
        reduced = _try_reduce_rank(U_best, V_best, W_best, n, m, p, max_iter=100)
        if reduced is not None:
            U_r, V_r, W_r, rank_r = reduced
            if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                best_rank = rank_r
                best_result = {"U": U_r, "V": V_r, "W": W_r}

    return best_result
