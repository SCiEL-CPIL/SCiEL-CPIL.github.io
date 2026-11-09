import time
import numpy as np
from scipy.linalg import solve

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    # Schoolbook rank
    rank_sb = n * m * p  # 42
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    col_indices = np.arange(rank_sb)
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat

    U_sb = np.zeros((nm, rank_sb), dtype=np.float64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.float64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.float64)
    U_sb[u_rows, col_indices] = 1
    V_sb[v_rows, col_indices] = 1
    W_sb[w_rows, col_indices] = 1

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap entries to nearest half-integer if close."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_solve_for_W(U, V, rank):
        """Given U and V, try to solve for W such that the decomposition holds."""
        W = np.zeros((pn, rank))
        valid = True

        for i in range(n):
            for k in range(p):
                # Build A_ik: m x rank matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]

                # Solve A_ik @ w = ones(m)
                try:
                    w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                        valid = False
                        break
                    W[k * n + i, :] = w
                except:
                    valid = False
                    break

        if not valid:
            return None
        return W

    def check_rank_deficiency(U, V):
        """Check for each (i,k) pair whether A_ik has rank <= m-1 = 2."""
        deficient = []
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m, nm))  # We'll build this properly
                A_ik = np.zeros((m, 1))  # placeholder
                # Actually build A_ik properly
                A_ik = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                # Check rank
                try:
                    r = np.linalg.matrix_rank(A_ik, tol=1e-8)
                    deficient.append(r <= m - 1)
                except:
                    deficient.append(False)
        return deficient

    def random_search_for_rank(rank, num_trials, entry_set, use_symmetry=False):
        """Random search with optional symmetry exploitation."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice(entry_set, size=(nm, rank))
            V = np.random.choice(entry_set, size=(mp, rank))

            if use_symmetry:
                # Exploit n=2 symmetry: make U[0:3,:] related to U[3:6,:]
                # and W[k*n,:] related to W[k*n+1,:]
                # Option 1: U[3:6,:] = U[0:3,:]
                if np.random.random() < 0.5:
                    U[3:6, :] = U[0:3, :].copy()
                else:
                    # Option 2: U[3:6,:] = -U[0:3,:]
                    U[3:6, :] = -U[0:3, :].copy()

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            # Try snapping to half-integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                # Try integer snap
                W_snapped = np.round(W)
                if not np.allclose(W, W_snapped, atol=1e-6):
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def structured_symmetry_search(rank, num_trials):
        """Search exploiting the n=2 symmetry more aggressively."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            # Generate base U_top (3 x rank) and V (21 x rank)
            U_top = np.random.choice([-1, 0, 1], size=(3, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))

            # Create full U with symmetry
            U = np.zeros((nm, rank))
            U[0:3, :] = U_top
            if np.random.random() < 0.5:
                U[3:6, :] = U_top  # symmetric
            else:
                U[3:6, :] = -U_top  # antisymmetric

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = np.round(W)
                if not np.allclose(W, W_snapped, atol=1e-6):
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def iterative_refinement_search(rank, num_trials):
        """Iteratively refine U,V to satisfy rank-deficiency constraints."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))

            # Iteratively check and adjust
            for iteration in range(50):
                if time.time() > deadline:
                    break

                deficient = check_rank_deficiency(U, V)
                if all(deficient):
                    break

                # Find the most violating (i,k) pair
                for i in range(n):
                    for k in range(p):
                        idx = i * p + k
                        if not deficient[idx]:
                            # Try to fix by adjusting one entry
                            j = np.random.randint(m)
                            if np.random.random() < 0.5:
                                U[i * m + j, np.random.randint(rank)] = np.random.choice([-1, 0, 1])
                            else:
                                V[j * p + k, np.random.randint(rank)] = np.random.choice([-1, 0, 1])
                            break
                    else:
                        continue
                    break

            # After refinement, try to solve for W
            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = np.round(W)
                if not np.allclose(W, W_snapped, atol=1e-6):
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    # Search for lower ranks
    # Priority: try symmetry-based searches first (most likely to succeed)
    for rank in range(rank_sb - 1, max(35, rank_sb - 10), -1):
        if time.time() > deadline:
            break

        # Try symmetry search
        U_found, V_found, W_found = structured_symmetry_search(rank, 50)
        if U_found is not None:
            best_U, best_V, best_W = U_found, V_found, W_found
            best_rank = rank
            break

    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(35, rank_sb - 10), -1):
            if time.time() > deadline:
                break

            # Try random search with symmetry
            U_found, V_found, W_found = random_search_for_rank(rank, 30, [-1, 0, 1], use_symmetry=True)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(35, rank_sb - 10), -1):
            if time.time() > deadline:
                break

            # Try iterative refinement
            U_found, V_found, W_found = iterative_refinement_search(rank, 20)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(30, rank_sb - 15), -1):
            if time.time() > deadline:
                break

            # Broader random search
            U_found, V_found, W_found = random_search_for_rank(rank, 20, [-1, 0, 1])
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

            U_found, V_found, W_found = random_search_for_rank(rank, 10, [0, 1])
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

    # Final verification and return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {"U": U_ret, "V": V_ret, "W": W_ret}
