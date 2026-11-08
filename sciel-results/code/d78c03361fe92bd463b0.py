import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150s kill
    n, m, p = DIMENSIONS

    # Target tensor: T[a,b,c] = 1 iff i==i' and j==j' and k==k'
    # a = i*3+j (A), b = j'*3+k (B), c = k'*3+i' (W, k*n+i convention)
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1
    # Precompute the "zero" entries (must stay 0) and "one" entries (must equal 1)
    zero_mask = (T == 0)
    one_mask = (T == 1)
    nz_idx = np.argwhere(T != 0)  # (27,3)

    def check(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)

    def get_rank26():
        rank = 26
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)

        col = 0
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1

        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1

        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1

        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1

        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1

        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1

        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1

        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1

        return U, V, W

    U26, V26, W26 = get_rank26()
    assert check(U26, V26, W26), "Rank-26 decomposition failed verification"

    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()

    # ------------------------------------------------------------------
    # Fast incremental-error simulated annealing.
    # error = sum_zero recon^2 + sum_one (recon-1)^2
    # A single element change only touches the slice along that axis.
    # ------------------------------------------------------------------
    def anneal(U, V, W, iters, time_limit, rng, temp0=8.0, cool=0.9995,
               delta_set=(-2, -1, 1, 2)):
        U = U.astype(np.int64).copy()
        V = V.astype(np.int64).copy()
        W = W.astype(np.int64).copy()
        rank = U.shape[1]

        def full_error():
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            e = int(np.sum((recon[zero_mask]) ** 2))
            d = recon[one_mask] - 1
            e += int(np.sum(d * d))
            return e

        cur_err = full_error()
        if cur_err == 0:
            return U, V, W, 0

        best_err = cur_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        temp = temp0
        no_improve = 0
        max_no_improve = 800

        for _ in range(iters):
            if time.time() > time_limit:
                break

            mat = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(rank)
            d = delta_set[rng.randint(len(delta_set))]

            if mat == 0:
                old = U[row, col]
                U[row, col] = old + d
                # slice: (a=row) -> over b,c for each r
                # delta = d * V[b,col] * W[c,col]
                delta = d * np.outer(V[:, col], W[:, col])
                oldv = recon_slice = (U[row, :, :] * V[:, :, col] * W[:, :, col])
                # recompute the row-slice contribution
                contrib_old = U[row, :, :] * V[:, :, col] * W[:, :, col]
                contrib_new = (U[row, :, :]) * V[:, :, col] * W[:, :, col]  # U already updated
                # Actually U[row] changed by d only at [col]; recompute slice:
                slice_old = (old) * V[:, col][:, None] * W[:, col][None, :]
                slice_new = (old + d) * V[:, col][:, None] * W[:, col][None, :]
                diff = slice_new - slice_old  # (9,9) over (b,c)
                # zero entries at a=row:
                z = zero_mask[row, :, :]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[row, :, :]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))
            elif mat == 1:
                old = V[row, col]
                V[row, col] = old + d
                slice_old = U[:, col][:, None] * (old) * W[:, col][None, :]
                slice_new = U[:, col][:, None] * (old + d) * W[:, col][None, :]
                diff = slice_new - slice_old
                z = zero_mask[:, row, :]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[:, row, :]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))
            else:
                old = W[row, col]
                W[row, col] = old + d
                slice_old = U[:, col][:, None] * V[:, col][:, None] * (old)
                slice_new = U[:, col][:, None] * V[:, col][:, None] * (old + d)
                z = zero_mask[:, :, row]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[:, :, row]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))

            if cur_err < 0:
                cur_err = 0

            accept = False
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                accept = True
                no_improve = 0
                if best_err == 0:
                    break
            elif cur_err < 0:
                accept = True
            else:
                # compare to pre-move error (need to store it)
                pass

            # We need the pre-move error for acceptance; recompute via reverting if not accepted.
            # Simpler: store prev_err before the move.
            # (handled below)
            temp *= cool
            no_improve += 1
            if no_improve > max_no_improve and temp < 0.5:
                break

        return best_U, best_V, best_W, best_err

    # The incremental logic above is error-prone; use a clean, correct version:
    def anneal2(U, V, W, iters, time_limit, rng, temp0=8.0, cool=0.9995,
                delta_set=(-2, -1, 1, 2)):
        U = U.astype(np.int64).copy()
        V = V.astype(np.int64).copy()
        W = W.astype(np.int64).copy()
        rank = U.shape[1]

        def full_error():
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            e = int(np.sum((recon[zero_mask]) ** 2))
            d = recon[one_mask] - 1
            e += int(np.sum(d * d))
            return e

        cur_err = full_error()
        if cur_err == 0:
            return U, V, W, 0

        best_err = cur_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        temp = temp0
        no_improve = 0
        max_no_improve = 800

        for _ in range(iters):
            if time.time() > time_limit:
                break

            mat = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(rank)
            d = delta_set[rng.randint(len(delta_set))]
            prev_err = cur_err

            if mat == 0:
                old = U[row, col]
                U[row, col] = old + d
                slice_old = old * V[:, col][:, None] * W[:, col][None, :]
                slice_new = (old + d) * V[:, col][:, None] * W[:, col][None, :]
                z = zero_mask[row, :, :]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[row, :, :]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))
            elif mat == 1:
                old = V[row, col]
                V[row, col] = old + d
                slice_old = U[:, col][:, None] * old * W[:, col][None, :]
                slice_new = U[:, col][:, None] * (old + d) * W[:, col][None, :]
                z = zero_mask[:, row, :]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[:, row, :]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))
            else:
                old = W[row, col]
                W[row, col] = old + d
                slice_old = U[:, col][:, None] * V[:, col][:, None] * old
                slice_new = U[:, col][:, None] * V[:, col][:, None] * (old + d)
                z = zero_mask[:, :, row]
                if z.any():
                    cur_err += int(np.sum((slice_old[z] ** 2) - (slice_new[z] ** 2)))
                o = one_mask[:, :, row]
                if o.any():
                    cur_err += int(np.sum((slice_new[o] - 1) ** 2 - (slice_old[o] - 1) ** 2))

            if cur_err < 0:
                cur_err = 0

            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve = 0
                if best_err == 0:
                    break
            elif cur_err <= prev_err:
                no_improve = 0
            else:
                # Metropolis accept?
                if temp > 1e-6 and rng.random() < np.exp(-(cur_err - prev_err) / temp):
                    no_improve = 0
                else:
                    # revert
                    if mat == 0:
                        U[row, col] = old
                    elif mat == 1:
                        V[row, col] = old
                    else:
                        W[row, col] = old
                    cur_err = prev_err

            temp *= cool
            no_improve += 1
            if no_improve > max_no_improve and temp < 0.5:
                break

        return best_U, best_V, best_W, best_err

    rng = np.random.RandomState(42)

    def full_error(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        e = int(np.sum((recon[zero_mask]) ** 2))
        d = recon[one_mask] - 1
        e += int(np.sum(d * d))
        return e

    # ------------------------------------------------------------------
    # Strategy A: column deletion + repair (reduce 26 -> 25 -> 24)
    # ------------------------------------------------------------------
    U_cur, V_cur, W_cur = U26.copy(), V26.copy(), W26.copy()
    current_rank = 26

    for target_rank in [25, 24]:
        if time.time() > deadline - 15:
            break
        success = False
        for attempt in range(25):
            if time.time() > deadline - 8:
                break
            col_to_remove = rng.randint(current_rank)
            idx = [c for c in range(current_rank) if c != col_to_remove]
            U_new = U_cur[:, idx]
            V_new = V_cur[:, idx]
            W_new = W_cur[:, idx]
            if full_error(U_new, V_new, W_new) == 0:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank = target_rank
                success = True
                break
            # repair with annealing
            U_s, V_s, W_s, e_s = anneal2(U_new, V_new, W_new,
                                         iters=4000, time_limit=deadline - 5,
                                         rng=rng, temp0=10.0)
            if e_s == 0:
                U_cur, V_cur, W_cur = U_s, V_s, W_s
                current_rank = target_rank
                success = True
                break
        if not success:
            break

    # ------------------------------------------------------------------
    # Strategy B: direct random-init low-rank search (inspiration idea)
    # ------------------------------------------------------------------
    for target_rank in [24, 23]:
        if time.time() > deadline - 10:
            break
        found = False
        for trial in range(15):
            if time.time() > deadline - 5:
                break
            U_t = rng.randint(-1, 2, size=(9, target_rank)).astype(np.int64)
            V_t = rng.randint(-1, 2, size=(9, target_rank)).astype(np.int64)
            W_t = rng.randint(-1, 2, size=(9, target_rank)).astype(np.int64)
            U_s, V_s, W_s, e_s = anneal2(U_t, V_t, W_t,
                                         iters=3000, time_limit=deadline - 5,
                                         rng=rng, temp0=50.0)
            if e_s == 0:
                U_cur, V_cur, W_cur = U_s, V_s, W_s
                current_rank = target_rank
                found = True
                break
        if found:
            continue
        else:
            break

    # ------------------------------------------------------------------
    # Final polish: one more annealing pass at the current rank to lock in
    # a clean exact solution, then verify.
    # ------------------------------------------------------------------
    if full_error(U_cur, V_cur, W_cur) != 0:
        U_cur, V_cur, W_cur = U26.copy(), V26.copy(), W26.copy()
        current_rank = 26

    # Polish (cheap, only if not already perfect)
    if full_error(U_cur, V_cur, W_cur) == 0:
        # already exact; keep
        pass

    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', U_cur, V_cur, W_cur)
    if not np.array_equal(recon, T):
        U_cur, V_cur, W_cur = U26, V26, W26
        current_rank = 26

    best_U, best_V, best_W = U_cur, V_cur, W_cur
    best_rank = current_rank

    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
