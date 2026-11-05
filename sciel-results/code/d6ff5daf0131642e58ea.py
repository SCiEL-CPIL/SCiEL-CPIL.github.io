"""
Improved approach: 
1. Find maximum number of disjoint 2x2x2 Strassen blocks via exhaustive search
2. Use a faster, more effective ALS refinement with better initialization
3. Try rank reduction via column elimination (removing redundant columns)
"""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    deadline = time.time() + 110  # safety margin

    # --- Step 1: Find maximum number of disjoint 2x2x2 Strassen blocks ---
    # All possible 2x2x2 block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))

    def blocks_disjoint(b1, b2):
        """Check if two 2x2x2 blocks are disjoint (share no (i,j,k) entry)."""
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # They're disjoint if at least one dimension has non-overlapping index sets
        # i sets: {i0a, i0a+1} vs {i0b, i0b+1}
        i_overlap = (i0a == i0b) or (i0a + 1 == i0b) or (i0b + 1 == i0a)
        j_overlap = (j0a == j0b) or (j0a + 1 == j0b) or (j0b + 1 == j0a)
        k_overlap = (k0a == k0b) or (k0a + 1 == k0b) or (k0b + 1 == k0a)
        # Disjoint if at least one dimension is non-overlapping
        return not (i_overlap and j_overlap and k_overlap)

    # Find maximum set of pairwise-disjoint blocks
    max_disjoint = []
    for size in range(min(len(all_blocks), 6), 0, -1):
        found = False
        for combo in combinations(all_blocks, size):
            disjoint = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if not blocks_disjoint(combo[i], combo[j]):
                        disjoint = False
                        break
                if not disjoint:
                    break
            if disjoint:
                max_disjoint = list(combo)
                found = True
                break
        if found:
            break

    num_strassen = len(max_disjoint)
    num_schoolbook = n * m * p - num_strassen * 8
    rank_deterministic = num_strassen * 7 + num_schoolbook

    # --- Step 2: Build the deterministic solution ---
    U = [[0.0] * rank_deterministic for _ in range(n * m)]
    V = [[0.0] * rank_deterministic for _ in range(m * p)]
    W = [[0.0] * rank_deterministic for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1.0
        U[i1 * m + j1][col_start] = 1.0
        V[j0 * p + k0][col_start] = 1.0
        V[j1 * p + k1][col_start] = 1.0
        W[k0 * n + i0][col_start] = 1.0
        W[k1 * n + i1][col_start] = 1.0

        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1.0
        U[i1 * m + j1][col_start + 1] = 1.0
        V[j0 * p + k0][col_start + 1] = 1.0
        W[k0 * n + i1][col_start + 1] = 1.0
        W[k1 * n + i1][col_start + 1] = -1.0

        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1.0
        V[j0 * p + k1][col_start + 2] = 1.0
        V[j1 * p + k1][col_start + 2] = -1.0
        W[k1 * n + i0][col_start + 2] = 1.0
        W[k1 * n + i1][col_start + 2] = 1.0

        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1.0
        V[j1 * p + k0][col_start + 3] = 1.0
        V[j0 * p + k0][col_start + 3] = -1.0
        W[k0 * n + i0][col_start + 3] = 1.0
        W[k0 * n + i1][col_start + 3] = 1.0

        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1.0
        U[i0 * m + j1][col_start + 4] = 1.0
        V[j1 * p + k1][col_start + 4] = 1.0
        W[k0 * n + i0][col_start + 4] = -1.0
        W[k1 * n + i0][col_start + 4] = 1.0

        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1.0
        U[i0 * m + j0][col_start + 5] = -1.0
        V[j0 * p + k0][col_start + 5] = 1.0
        V[j0 * p + k1][col_start + 5] = 1.0
        W[k1 * n + i1][col_start + 5] = 1.0

        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1.0
        U[i1 * m + j1][col_start + 6] = -1.0
        V[j1 * p + k0][col_start + 6] = 1.0
        V[j1 * p + k1][col_start + 6] = 1.0
        W[k0 * n + i0][col_start + 6] = 1.0

        return col_start + 7

    # Add Strassen blocks
    for (i0, j0, k0) in max_disjoint:
        col = add_strassen_block(col, i0, j0, k0)

    # Mark covered entries
    covered = set()
    for (i0, j0, k0) in max_disjoint:
        for i in (i0, i0 + 1):
            for j in (j0, j0 + 1):
                for k in (k0, k0 + 1):
                    covered.add((i, j, k))

    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1

    assert col == rank_deterministic, f"Expected rank {rank_deterministic}, got {col}"

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    def check_solution(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i * m + j, :], V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    assert check_solution(U_np, V_np, W_np, tol=1e-9), "Deterministic solution failed verification"

    best_rank = rank_deterministic
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

    # --- Step 3: Column elimination - try to remove redundant columns ---
    # A column is redundant if removing it still leaves a valid decomposition
    # (i.e., the remaining columns can be adjusted to still work)
    
    def try_column_elimination(U_mat, V_mat, W_mat, max_cols_to_remove=5):
        """Try to remove columns one at a time, checking validity after each removal."""
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        current_rank = U_cur.shape[1]
        
        removed = 0
        for attempt in range(max_cols_to_remove):
            if time.time() > deadline:
                break
            found_redundant = False
            for c in range(current_rank):
                if time.time() > deadline:
                    break
                # Try removing column c
                mask = [i for i in range(current_rank) if i != c]
                U_test = U_cur[:, mask]
                V_test = V_cur[:, mask]
                W_test = W_cur[:, mask]
                
                if check_solution(U_test, V_test, W_test, tol=1e-8):
                    # Column c is redundant
                    U_cur = U_test
                    V_cur = V_test
                    W_cur = W_test
                    current_rank -= 1
                    removed += 1
                    found_redundant = True
                    break
            
            if not found_redundant:
                break
        
        return U_cur, V_cur, W_cur, removed

    U_elim, V_elim, W_elim, num_removed = try_column_elimination(U_np, V_np, W_np, max_cols_to_remove=10)
    if num_removed > 0:
        best_rank = rank_deterministic - num_removed
        best_U, best_V, best_W = U_elim, V_elim, W_elim

    # --- Step 4: Fast ALS refinement ---
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    min_rank = max(n * m, m * p, p * n)

    def fast_als_decompose(rank, max_iter=50, seed=0):
        """
        Fast ALS with structured initialization from Strassen pattern.
        Uses vectorized operations for speed.
        """
        np.random.seed(seed)

        # Initialize from Strassen structure
        if rank >= U_np.shape[1]:
            U_init = U_np[:, :rank].copy()
            V_init = V_np[:, :rank].copy()
            W_init = W_np[:, :rank].copy()
        else:
            # Take a subset and add perturbation
            U_init = U_np[:, :rank].copy()
            V_init = V_np[:, :rank].copy()
            W_init = W_np[:, :rank].copy()
            U_init += np.random.randn(n * m, rank) * 0.1
            V_init += np.random.randn(m * p, rank) * 0.1
            W_init += np.random.randn(p * n, rank) * 0.1

        U_cur = U_init
        V_cur = V_init
        W_cur = W_init

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None

            # Update U: for each (i,j), solve for U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    # A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                    # We need sum_r U[i*m+j, r] * A[r, k] = 1 for all k
                    A = np.array([V_cur[j * p + k, :] * W_cur[k * n + i, :] for k in range(p)])
                    # A shape: (p, rank)
                    try:
                        U_cur[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        pass

            # Update V: for each (j,k), solve for V[j*p+k,:]
            for j in range(m):
                for k in range(p):
                    A = np.array([U_cur[i * m + j, :] * W_cur[k * n + i, :] for i in range(n)])
                    # A shape: (n, rank)
                    try:
                        V_cur[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        pass

            # Update W: for each (k,i), solve for W[k*n+i,:]
            for k in range(p):
                for i in range(n):
                    A = np.array([U_cur[i * m + j, :] * V_cur[j * p + k, :] for j in range(m)])
                    # A shape: (m, rank)
                    try:
                        W_cur[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        pass

            # Check validity
            if check_solution(U_cur, V_cur, W_cur, tol=1e-4):
                # Snap to half-integers
                U_snap = np.round(U_cur * 2) / 2
                V_snap = np.round(V_cur * 2) / 2
                W_snap = np.round(W_cur * 2) / 2
                if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap

        # Final check
        if check_solution(U_cur, V_cur, W_cur, tol=1e-8):
            U_snap = np.round(U_cur * 2) / 2
            V_snap = np.round(V_cur * 2) / 2
            W_snap = np.round(W_cur * 2) / 2
            if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                return U_snap, V_snap, W_snap

        return None

    # Try ranks from lower bound up to current best
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break

        found = False
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            seed = (rank * 1000 + restart) % (2**31)
            result = fast_als_decompose(rank, max_iter=40, seed=seed)
            if result is not None:
                U_f, V_f, W_f = result
                best_rank = rank
                best_U, best_V, best_W = U_f, V_f, W_f
                found = True
                break

        if found:
            break

    # --- Step 5: Final safety check ---
    if not check_solution(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W = U_np, V_np, W_np
        best_rank = rank_deterministic

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
