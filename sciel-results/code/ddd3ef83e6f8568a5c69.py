import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_decomposition(U, V, W, rank):
        """Vectorized check using einsum over all columns at once."""
        U3 = U.reshape(n, m, rank)  # U[i,j,c]
        V3 = V.reshape(m, p, rank)  # V[j,k,c]
        W3 = W.reshape(p, n, rank)  # W[k,i,c]
        # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
        return np.all(result == 1)

    def check_decomposition_partial(U, V, W, rank):
        """Returns the max absolute error from 1, for gradient-free search."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True).astype(np.float64)
        return np.max(np.abs(result - 1.0))

    def kronecker_init(rank, attempt_idx):
        """
        Exploit symmetry between m and p (both 5).
        Use Kronecker products of identity/permutation blocks.
        The idea: for 3x5x5, we can think of the 5-dim indices as
        (block, within-block) and use structured rank factors.
        """
        np.random.seed(777 + attempt_idx * 421 + rank * 13)

        if attempt_idx % 4 == 0:
            # Block-diagonal structure: treat 5 as 2+3 or use I_5 kron patterns
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            # Use permutation matrices and small random perturbations
            for c in range(rank):
                # Random permutation-based initialization
                perm_u = np.random.permutation(n * m)
                perm_v = np.random.permutation(m * p)
                perm_w = np.random.permutation(p * n)
                # Place ±1 at a few positions
                n_u = random.randint(2, 4)
                n_v = random.randint(2, 4)
                n_w = random.randint(2, 4)
                for idx in perm_u[:n_u]:
                    U[idx, c] = random.choice([-1, 1])
                for idx in perm_v[:n_v]:
                    V[idx, c] = random.choice([-1, 1])
                for idx in perm_w[:n_w]:
                    W[idx, c] = random.choice([-1, 1])
        elif attempt_idx % 4 == 1:
            # Kronecker: I_5 ⊗ A pattern for the 5-dim parts
            # U has shape (3*5, rank), think of it as (3, 5, rank)
            # For each "block" of 5, use structured patterns
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            for c in range(rank):
                # Random ±1 pattern with some structure
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.5:
                            U[i * m + j, c] = random.choice([-1, 1])
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j * p + k, c] = random.choice([-1, 1])
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.5:
                            W[k * n + i, c] = random.choice([-1, 1])
        elif attempt_idx % 4 == 2:
            # Symmetric initialization: V and W share structure (m=p=5)
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            for c in range(rank):
                # Generate V, then derive W from V using symmetry
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j * p + k, c] = random.choice([-1, 1])
                # W[k*n+i] mirrors V[j*p+k] structure (j<->k swap)
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            W[k * n + 0, c] = random.choice([-1, 1])  # i=0
                        if random.random() < 0.5:
                            W[k * n + 1, c] = random.choice([-1, 1])  # i=1
                        if random.random() < 0.5:
                            W[k * n + 2, c] = random.choice([-1, 1])  # i=2
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.5:
                            U[i * m + j, c] = random.choice([-1, 1])
        else:
            # Dense random with values in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))

        return U, V, W

    def local_search(U, V, W, rank, max_iters=5000, time_budget=5.0):
        """
        Coordinate descent: try flipping individual entries to reduce max error.
        Uses the partial check to evaluate improvements.
        """
        ts = time.time()
        current_err = check_decomposition_partial(U, V, W, rank)
        if current_err == 0:
            return U, V, W, True

        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)

        # Precompute for fast updates
        # result[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # If we change U3[i0,j0,c0] by delta, result changes by:
        #   delta * V3[j0,k,c0] * W3[k,i0,c0] for all i0,j0,k
        # Actually: result[i,j,k] += delta * V3[j,k,c0] * W3[k,i,c0] only for the changed (i,j)

        # For efficiency, precompute per-column contributions
        # contrib_c[i,j,k] = V3[j,k,c] * W3[k,i,c]
        # Then result[i,j,k] = sum_c U3[i,j,c] * contrib_c[i,j,k]

        # Build contrib array: shape (rank, n, m, p)
        # contrib[c, i, j, k] = V3[j,k,c] * W3[k,i,c]
        # = V3[:, :, c].T reshaped... let's compute:
        # V3[j,k,c] has shape (m,p,rank) -> transpose to (rank,m,p)
        # W3[k,i,c] has shape (p,n,rank) -> transpose to (rank,p,n)
        # contrib[c,i,j,k] = V3_t[c,j,k] * W3_t[c,k,i]
        V3_t = V3.transpose(2, 0, 1)  # (rank, m, p)
        W3_t = W3.transpose(2, 0, 1)  # (rank, p, n)
        # contrib[c, i, j, k] = V3_t[c, j, k] * W3_t[c, k, i]
        # Shape: for each c, V3_t[c] is (m,p), W3_t[c] is (p,n)
        # We need (m,p) * (p,n) -> but we want (i,j,k) indexing
        # contrib[c, i, j, k] = V3_t[c, j, k] * W3_t[c, k, i]
        # This is: for fixed c, outer product of V3_t[c] (m,p) and W3_t[c] (p,n) with index alignment
        # Let's compute: contrib[c] = np.einsum('jk,ki->ijk', V3_t[c], W3_t[c])
        # But that's (n,m,p) for each c -> shape (rank, n, m, p)
        contrib = np.zeros((rank, n, m, p), dtype=np.int64)
        for c in range(rank):
            contrib[c] = np.einsum('jk,ki->ijk', V3_t[c], W3_t[c], optimize=True)

        # result[i,j,k] = sum_c U3[i,j,c] * contrib[c,i,j,k]
        # = sum_c U3[i,j,c] * contrib[c,i,j,k]

        best_err = current_err
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()

        for it in range(max_iters):
            if time.time() - ts > time_budget:
                break
            if best_err == 0:
                return U_work, V_work, True

            # Try flipping a random entry
            # Pick random matrix and position
            mat_idx = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if mat_idx == 0:
                pos = (random.randint(0, n * m - 1), random.randint(0, rank - 1))
                i0, j0 = pos[0] // m, pos[0] % m
                c0 = pos[1]
                old_val = U_work[pos[0], pos[1]]
                # Try delta = -2*old_val (flip sign) or +1, -1
                deltas = [d for d in [-2, -1, 1, 2] if old_val + d != old_val]
                best_delta = 0
                best_new_err = best_err
                for d in deltas:
                    # Update: result[i0,j0,k] += d * contrib[c0, i0, j0, k]
                    # New error:
                    # Compute new result for affected entries
                    # Only (i0, j0, *) changes
                    old_result_row = np.einsum('c,cijk->k', U_work.reshape(n, m, rank)[i0, j0, :], contrib[:, i0, j0, :], optimize=True)
                    new_result_row = old_result_row + d * contrib[c0, i0, j0, :]
                    # Error for this row:
                    new_err_row = np.max(np.abs(new_result_row - 1.0))
                    # Also need to check other (i,j) pairs are unaffected, so max error is max(best_err_other, new_err_row)
                    # But best_err might be in this row or elsewhere
                    # Simplification: just compute full error if this looks promising
                    if new_err_row < best_new_err or best_err > 0.5:
                        # Compute full error
                        U_temp = U_work.copy()
                        U_temp[pos[0], pos[1]] = old_val + d
                        new_err = check_decomposition_partial(U_temp, V_work, W_work, rank)
                        if new_err < best_new_err:
                            best_new_err = new_err
                            best_delta = d
                if best_delta != 0:
                    U_work[pos[0], pos[1]] = old_val + best_delta
                    if best_new_err < best_err:
                        best_err = best_new_err
                        # Update contrib: contrib[c0, i0, j0, :] changes
                        # contrib[c0, i, j, k] = V3_t[c0, j, k] * W3_t[c0, k, i]
                        # Only contrib[c0, i0, j0, :] is affected by U change? No!
                        # contrib doesn't depend on U. So no update needed.
            elif mat_idx == 1:
                pos = (random.randint(0, m * p - 1), random.randint(0, rank - 1))
                j0, k0 = pos[0] // p, pos[0] % p
                c0 = pos[1]
                old_val = V_work[pos[0], pos[1]]
                deltas = [d for d in [-2, -1, 1, 2] if old_val + d != old_val]
                best_delta = 0
                best_new_err = best_err
                for d in deltas:
                    # Changing V3[j0,k0,c0] by d affects:
                    # result[i,j0,k] for all i,k where W3[k,i,c0] != 0
                    # And contrib[c0, i, j0, k] changes for all i,k
                    # New contrib[c0, i, j0, k] += d * W3_t[c0, k, i]
                    # So result[i, j0, k] += U3[i, j0, c0] * d * W3_t[c0, k, i]
                    # = d * U3[i, j0, c0] * W3_t[c0, k, i]
                    # Let's just compute full error
                    V_temp = V_work.copy()
                    V_temp[pos[0], pos[1]] = old_val + d
                    new_err = check_decomposition_partial(U_work, V_temp, W_work, rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_delta = d
                if best_delta != 0:
                    V_work[pos[0], pos[1]] = old_val + best_delta
                    if best_new_err < best_err:
                        best_err = best_new_err
            else:
                pos = (random.randint(0, p * n - 1), random.randint(0, rank - 1))
                k0, i0 = pos[0] // n, pos[0] % n
                c0 = pos[1]
                old_val = W_work[pos[0], pos[1]]
                deltas = [d for d in [-2, -1, 1, 2] if old_val + d != old_val]
                best_delta = 0
                best_new_err = best_err
                for d in deltas:
                    W_temp = W_work.copy()
                    W_temp[pos[0], pos[1]] = old_val + d
                    new_err = check_decomposition_partial(U_work, V_work, W_temp, rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_delta = d
                if best_delta != 0:
                    W_work[pos[0], pos[1]] = old_val + best_delta
                    if best_new_err < best_err:
                        best_err = best_new_err

        return U_work, V_work, W_work, (best_err == 0)

    # Search from high rank down
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None

    # Try a range of ranks
    ranks_to_try = list(range(74, 55, -1)) + list(range(55, 40, -3))

    for rank in ranks_to_try:
        if time.time() > deadline:
            break

        # Scale attempts based on time remaining
        time_left = deadline - time.time()
        if time_left < 5:
            break
        max_attempts = min(20, int(time_left / 3))
        max_attempts = max(max_attempts, 3)

        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            U, V, W = kronecker_init(rank, attempt)

            # Quick check
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break

            # Try local search on near-misses
            err = check_decomposition_partial(U, V, W, rank)
            if err < 2.0 and time.time() < deadline - 10:
                U2, V2, W2, success = local_search(U, V, W, rank, max_iters=3000, time_budget=3.0)
                if success:
                    best_rank = rank
                    best_U, best_V, best_W = U2, V2, W2
                    found = True
                    break

        if found:
            # Found something, try to go lower
            if rank <= 50:
                break

    if best_rank < n * m * p and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W, _ = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
