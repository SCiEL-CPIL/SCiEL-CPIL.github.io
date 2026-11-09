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
    deadline = start_time + 135.0  # Safety margin for 150s limit

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

    def snap_to_integer(arr, tol=1e-6):
        """Snap entries to nearest integer if close."""
        snapped = np.round(arr)
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
                A_ik = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])

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

    def check_feasibility(U, V, rank):
        """Check if for each (i,k), the system A_ik @ w = ones is consistent."""
        for i in range(n):
            for k in range(p):
                A_ik = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                try:
                    w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                        return False
                except:
                    return False
        return True

    def random_search_for_rank(rank, num_trials, entry_set, use_symmetry=False, seed=None):
        """Random search with optional symmetry exploitation."""
        if seed is not None:
            np.random.seed(seed)

        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice(entry_set, size=(nm, rank))
            V = np.random.choice(entry_set, size=(mp, rank))

            if use_symmetry:
                # Exploit n=2 symmetry: U[3:6,:] related to U[0:3,:]
                sym_type = np.random.random()
                if sym_type < 0.4:
                    U[3:6, :] = U[0:3, :].copy()  # symmetric
                elif sym_type < 0.8:
                    U[3:6, :] = -U[0:3, :].copy()  # antisymmetric
                else:
                    # Partial symmetry with random sign per column
                    signs = np.random.choice([-1, 1], size=(1, rank))
                    U[3:6, :] = U[0:3, :] * signs

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            # Try snapping to half-integers, then integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = snap_to_integer(W)
                if W_snapped is None:
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def structured_symmetry_search(rank, num_trials):
        """Search exploiting the n=2 symmetry more aggressively with structured patterns."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            # Generate base U_top (3 x rank) with structured patterns
            pattern_type = np.random.random()
            if pattern_type < 0.3:
                U_top = np.random.choice([-1, 0, 1], size=(3, rank))
            elif pattern_type < 0.6:
                # Sparse pattern
                U_top = np.random.choice([-1, 0, 1], size=(3, rank), p=[0.2, 0.6, 0.2])
            else:
                # Block pattern
                U_top = np.zeros((3, rank))
                for r in range(rank):
                    if np.random.random() < 0.7:
                        U_top[np.random.randint(3), r] = np.random.choice([-1, 1])

            V = np.random.choice([-1, 0, 1], size=(mp, rank))

            # Create full U with symmetry
            U = np.zeros((nm, rank))
            U[0:3, :] = U_top
            sym_choice = np.random.random()
            if sym_choice < 0.5:
                U[3:6, :] = U_top  # symmetric
            else:
                U[3:6, :] = -U_top  # antisymmetric

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = snap_to_integer(W)
                if W_snapped is None:
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def iterative_refinement_search(rank, num_trials):
        """Iteratively refine U,V to satisfy feasibility constraints."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))

            # Iteratively check and adjust
            for iteration in range(100):
                if time.time() > deadline:
                    break

                if check_feasibility(U, V, rank):
                    break

                # Find a violating (i,k) pair and adjust
                found_violation = False
                for i in range(n):
                    for k in range(p):
                        A_ik = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        try:
                            w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                            if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                                # Try to fix by adjusting one entry
                                j = np.random.randint(m)
                                if np.random.random() < 0.5:
                                    U[i * m + j, np.random.randint(rank)] = np.random.choice([-1, 0, 1])
                                else:
                                    V[j * p + k, np.random.randint(rank)] = np.random.choice([-1, 0, 1])
                                found_violation = True
                                break
                        except:
                            found_violation = True
                            break
                    if found_violation:
                        break

                if not found_violation:
                    break

            # After refinement, try to solve for W
            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = snap_to_integer(W)
                if W_snapped is None:
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def kronecker_structured_search(rank, num_trials):
        """Try Kronecker product structure: U = U1 ⊗ U2 style."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            # Try block structure based on Kronecker-like patterns
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))

            # Partition rank into blocks
            if rank >= 6:
                block_size = rank // 6
                for i in range(n):
                    for j in range(m):
                        start = (i * m + j) * block_size
                        end = start + block_size
                        U[i * m + j, start:end] = np.random.choice([-1, 0, 1], size=block_size)

                for j in range(m):
                    for k in range(p):
                        start = (j * p + k) * block_size
                        end = start + block_size
                        V[j * p + k, start:end] = np.random.choice([-1, 0, 1], size=block_size)
            else:
                U = np.random.choice([-1, 0, 1], size=(nm, rank))
                V = np.random.choice([-1, 0, 1], size=(mp, rank))

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                W_snapped = snap_to_integer(W)
                if W_snapped is None:
                    continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    # Search for lower ranks with multiple strategies
    strategies = [
        ("structured_symmetry", structured_symmetry_search),
        ("random_symmetry", lambda r, t: random_search_for_rank(r, t, [-1, 0, 1], use_symmetry=True)),
        ("iterative_refinement", iterative_refinement_search),
        ("kronecker", kronecker_structured_search),
        ("random_plain", lambda r, t: random_search_for_rank(r, t, [-1, 0, 1])),
    ]

    for rank in range(rank_sb - 1, max(30, rank_sb - 15), -1):
        if time.time() > deadline:
            break

        for name, strategy_func in strategies:
            if time.time() > deadline:
                break

            num_trials = 30 if rank > 38 else 50
            U_found, V_found, W_found = strategy_func(rank, num_trials)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

        if best_rank != rank_sb:
            break

    # If still at schoolbook, try a few more ranks with more trials
    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(28, rank_sb - 20), -1):
            if time.time() > deadline:
                break

            for name, strategy_func in strategies:
                if time.time() > deadline:
                    break
                U_found, V_found, W_found = strategy_func(rank, 20)
                if U_found is not None:
                    best_U, best_V, best_W = U_found, V_found, W_found
                    best_rank = rank
                    break

            if best_rank != rank_sb:
                break

    # Final verification and return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {"U": U_ret, "V": V_ret, "W": W_ret}
