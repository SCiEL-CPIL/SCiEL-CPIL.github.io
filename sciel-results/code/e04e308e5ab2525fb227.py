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


def _verify_decomposition_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy arrays directly (faster when already in numpy form)."""
    rank = U.shape[2]
    if rank == 0:
        return False
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U, V, W)
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


def _try_reduce_rank_aggressive(U, V, W, n, m, p, max_iter=500, tol=1e-9, deadline=None):
    """Aggressive rank reduction: try removing single terms, pairs, and triples."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if deadline is None:
        deadline = time.time() + 100
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        removed = False
        
        # Phase 1: Try removing individual terms (fastest)
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
        
        if removed:
            continue
        
        # Phase 2: Try removing pairs (more expensive but can find synergies)
        if rank > 2 and iteration % 2 == 0:
            for c1 in range(rank - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                for c2 in range(c1 - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    idxs = [i for i in range(rank) if i != c1 and i != c2]
                    U_test = [[U[i][j] for j in idxs] for i in range(n * m)]
                    V_test = [[V[i][j] for j in idxs] for i in range(m * p)]
                    W_test = [[W[i][j] for j in idxs] for i in range(p * n)]
                    if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                        U = U_test
                        V = V_test
                        W = W_test
                        rank -= 2
                        removed = True
                        break
                if removed:
                    break
        
        if not removed:
            break

    return U, V, W, rank


def _try_linear_combination_removal(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Try to find linear dependencies among rank-1 terms that allow reduction.
    If term c1 = alpha * term c2 (in all three matrices), we can remove c1."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if deadline is None:
        deadline = time.time() + 50
    
    for iteration in range(rank):
        if time.time() > deadline - 5:
            break
        
        removed = False
        for c1 in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            for c2 in range(rank - 1, c1, -1):
                # Check if U[:,c1] is proportional to U[:,c2]
                # Find alpha from first non-zero element
                alpha = None
                for idx in range(n * m):
                    if abs(U[idx][c2]) > tol:
                        alpha = U[idx][c1] / U[idx][c2]
                        break
                
                if alpha is None:
                    continue
                
                # Verify proportionality
                match = True
                for idx in range(n * m):
                    if abs(U[idx][c1] - alpha * U[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue
                
                for idx in range(m * p):
                    if abs(V[idx][c1] - alpha * V[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue
                
                for idx in range(p * n):
                    if abs(W[idx][c1] - alpha * W[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue
                
                # Remove c1
                U_test = [row[:c1] + row[c1 + 1:] for row in U]
                V_test = [row[:c1] + row[c1 + 1:] for row in V]
                W_test = [row[:c1] + row[c1 + 1:] for row in W]
                if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    removed = True
                    break
            if removed:
                break
        
        if not removed:
            break

    return U, V, W, rank


def _build_4x4x5_compact():
    """Try to build a more compact 4x4x5 decomposition.
    
    Key insight: The 5th column of B can be handled by modifying the Strassen
    core terms rather than adding 16 separate boundary terms.
    
    Strategy: Use the 4x4x4 Strassen structure but modify it so that the
    5th column contribution is embedded in the existing 28 terms, plus
    a small number of additional terms.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    
    # We need to handle the 5th column: C[:,4] = A @ B[:,4]
    # For the 4x4x4 part, we have sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for k=0..3
    # For the 5th column: sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1 for all i,j
    
    # If we set V[j,4,c] and W[4,i,c] appropriately, we can reuse the U terms.
    # The simplest approach: add terms where only the 5th column is affected.
    
    # Alternative: Try to find a rank-r decomposition where r < 44
    # by using a different block structure.
    
    # For now, return the standard construction
    return _build_4x4x5_from_strassen()


def _search_low_rank(n, m, p, target_rank, deadline):
    """Search for a decomposition of the given target rank using structured patterns."""
    start = time.time()
    
    # Try different structured patterns
    # Pattern 1: Diagonal + off-diagonal terms
    # Pattern 2: Block-diagonal with shared terms
    
    # For 4x4x5, the minimum possible rank is at least max(4*4, 4*5, 5*4) = 20
    # by the trivial lower bound (each term can contribute to at most one (i,j,k) 
    # in a "diagonal" sense, but this isn't tight).
    
    # Actually, the lower bound is more subtle. Let's try various target ranks.
    
    # Use a greedy approach: build up terms one at a time, trying to satisfy
    # as many (i,j,k) constraints as possible.
    
    best_U = None
    best_V = None
    best_W = None
    
    # Try random sparse patterns
    import random
    rng = random.Random(42)
    
    max_attempts = min(50, int((deadline - start) / 0.1))
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Randomly assign sparse values
        for c in range(target_rank):
            # Each term should be somewhat sparse
            for idx in range(n * m):
                if rng.random() < 0.2:
                    U[idx][c] = rng.choice([-1, 1])
            for idx in range(m * p):
                if rng.random() < 0.2:
                    V[idx][c] = rng.choice([-1, 1])
            for idx in range(p * n):
                if rng.random() < 0.2:
                    W[idx][c] = rng.choice([-1, 1])
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

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

            # Aggressive rank reduction
            if time.time() < deadline - 10:
                U_r, V_r, W_r, rank_r = _try_reduce_rank_aggressive(
                    U_new, V_new, W_new, n, m, p, 
                    max_iter=500, deadline=deadline - 5
                )
                if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                    best_rank = rank_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}

            # Try linear combination removal
            if time.time() < deadline - 5:
                U_r, V_r, W_r, rank_r = _try_linear_combination_removal(
                    best_result["U"], best_result["V"], best_result["W"],
                    n, m, p, deadline=deadline - 2
                )
                if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                    best_rank = rank_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}

    # Try to find lower-rank solutions via structured search
    if time.time() < deadline - 10 and best_rank > 30:
        # Search for lower rank decompositions
        for target_rank in range(28, best_rank):
            if time.time() > deadline - 10:
                break
            
            result = _search_low_rank(n, m, p, target_rank, deadline - 10)
            if result is not None:
                U_f, V_f, W_f = result
                if _verify_decomposition_fast(U_f, V_f, W_f, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_f, "V": V_f, "W": W_f}
                    break

    # Final reduction pass
    if time.time() < deadline - 5:
        U_best, V_best, W_best = best_result["U"], best_result["V"], best_result["W"]
        U_r, V_r, W_r, rank_r = _try_reduce_rank_aggressive(
            U_best, V_best, W_best, n, m, p,
            max_iter=200, deadline=deadline - 2
        )
        if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
            best_rank = rank_r
            best_result = {"U": U_r, "V": V_r, "W": W_r}

    return best_result
