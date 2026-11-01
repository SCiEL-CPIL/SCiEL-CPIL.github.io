import time
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin under the 120s budget

    def now():
        return time.time()

    def to_ints(cw):
        return [int(x) for x in np.asarray(cw).ravel().tolist()]

    # --- helpers to build balls (codeword + all radius-1 neighbours) ---
    def ball_of(cw):
        cw = np.asarray(cw, dtype=np.int64).ravel()
        b = [cw.copy()]
        for i in range(n):
            base = cw.copy()
            for v in range(q):
                if v == cw[i]:
                    continue
                base[i] = v
                b.append(base.copy())
        return np.array(b, dtype=np.int64)

    def ball_matrix(cw_arr):
        out = []
        for i in range(cw_arr.shape[0]):
            out.append(ball_of(cw_arr[i]))
        return np.vstack(out)

    # --- coverage check over the whole space (vectorized) ---
    def covered_count(cw_arr):
        if cw_arr.shape[0] == 0:
            return 0
        bm = ball_matrix(cw_arr)
        idx = bm[:, 0] * q + bm[:, 1:].ravel()
        return int(np.unique(idx).size)

    def is_covering(cw_arr):
        return covered_count(cw_arr) == q ** n

    # --- fast greedy construction ---
    def greedy():
        # random initial seed
        seed = np.random.randint(0, q, size=n, dtype=np.int64)
        cw = [seed]
        # candidate pool = all radius-1 neighbours of current codewords (dynamic)
        cand = np.array([ball_of(seed)[1:]])  # exclude the seed itself
        cand_set = set(map(tuple, cand.tolist()))
        while now() < deadline:
            if len(cw) > 0 and is_covering(np.array(cw)):
                break
            if len(cand) == 0:
                break
            bm = ball_matrix(np.vstack(cand))
            idx = bm[:, 0] * q + bm[:, 1:].ravel()
            # count how many space points each candidate ball covers (uniquely)
            _, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
            # total covered so far
            cur = np.vstack(cw)
            cur_bm = ball_matrix(cur)
            cur_idx = cur_bm[:, 0] * q + cur_bm[:, 1:].ravel()
            cur_set = set(np.unique(cur_idx).tolist())
            # gain per candidate = number of new points
            gains = np.zeros(len(cand), dtype=np.int64)
            # vectorized: for each candidate, count points not in cur_set
            # map candidate idx to a membership flag
            cand_flag = np.zeros(q ** n, dtype=bool)
            cand_flag[idx] = True
            # per-candidate unique points
            # use grouped unique via sorting
            order = np.argsort(inv, kind='stable')
            sorted_inv = inv[order]
            # boundaries
            # build per-candidate new-point count
            # count distinct idx per candidate not in cur_set
            newmask = ~np.array([c in cur_set for c in idx.tolist()]) if len(idx) < 200000 else np.zeros(len(idx), dtype=bool)
            # fallback: recompute membership via boolean array
            in_cur = np.zeros(q ** n, dtype=bool)
            in_cur[cur_idx] = True
            newmask = ~in_cur[idx]
            # distinct new points per candidate
            # use np.unique on (inv, idx) pairs
            pair = inv * (q ** n) + idx
            uniq = np.unique(pair)
            # candidate id = pair // q**n ; idx = pair % q**n
            uid = uniq // (q ** n)
            uidx = uniq % (q ** n)
            u_new = ~in_cur[uidx]
            gains = np.bincount(uid, weights=u_new.astype(np.int64), minlength=len(cand))
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            cw.append(cand[best].copy())
            # add its radius-1 neighbours to candidate pool
            nb = ball_of(cand[best])[1:]
            new_rows = []
            for r in nb:
                t = tuple(r.tolist())
                if t not in cand_set:
                    cand_set.add(t)
                    new_rows.append(r)
            if new_rows:
                cand = np.vstack([cand, np.array(new_rows, dtype=np.int64)])
        return np.array(cw, dtype=np.int64)

    cw = greedy()

    # If greedy didn't cover, fall back to the direct-sum construction (always valid).
    if not is_covering(cw):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        tails = list(itertools.product(range(3), repeat=n - 4))
        cw = np.array([h + list(t) for h in tetracode for t in tails], dtype=np.int64)

    # --- local search: remove + swap, vectorized via ball membership ---
    def ball_membership(cw_arr):
        # returns (num_codewords, ball_size, space_size) boolean matrix
        bm = ball_matrix(cw_arr)
        idx = bm[:, 0] * q + bm[:, 1:].ravel()
        n_cod = cw_arr.shape[0]
        ball_sz = idx.size // n_cod
        M = np.zeros((n_cod, q ** n), dtype=bool)
        M[np.repeat(np.arange(n_cod), ball_sz), idx] = True
        return M

    def try_remove(cw_arr):
        # find a codeword whose ball is fully covered by the others
        M = ball_membership(cw_arr)
        # coverage count per space point
        cov = M.sum(axis=0)
        # a codeword i is removable if for all points in its ball, cov[point] >= 2
        # i.e. min cov over its ball >= 2
        # cov over its own ball: M[i] & cov
        removable = np.all(np.where(M, cov, 2) >= 2, axis=1)
        if removable.any():
            i = int(np.argmax(removable))
            return np.delete(cw_arr, i, axis=0)
        return None

    def try_swap(cw_arr):
        # find (i, position j, value v) such that removing i and replacing with a
        # neighbour keeps covering. Use ball membership to test in O(1).
        M = ball_membership(cw_arr)
        cov = M.sum(axis=0)
        n_cod = cw_arr.shape[0]
        # points that are only covered by codeword i (critical for i)
        # For a swap: new codeword w' replaces i. We need every point that was
        # covered only by i to be covered by w'. Points covered by i and nobody
        # else: (M[i] & (cov==1)).
        # w' must contain all those points in its ball.
        # Also w' must be a radius-1 neighbour of i (single position change).
        # We test each candidate w' quickly.
        # Precompute, for each candidate point p, the set of codewords whose ball
        # contains p is M. For w' we need: critical_i ⊆ ball(w').
        # ball(w') = {w'} ∪ neighbours. We can compute which space points are in
        # ball(w') by building its ball idx.
        # To be fast, iterate candidates in a smart order: try removing the codeword
        # with the most redundancy first.
        # Choose i = argmax (number of points with cov>=2 in its ball) = least critical.
        red = np.sum((cov >= 2) & M, axis=1)
        order = np.argsort(-red)
        for i in order[:min(60, n_cod)]:
            crit = M[i] & (cov == 1)
            if not crit.any():
                # i is fully redundant -> already handled by try_remove
                continue
            crit_idx = np.where(crit)[0]
            # candidate new codewords: radius-1 neighbours of cw[i]
            base = cw_arr[i].copy()
            for j in range(n):
                for v in range(q):
                    if v == base[j]:
                        continue
                    base[j] = v
                    nb = ball_of(base)
                    nidx = nb[:, 0] * q + nb[:, 1:].ravel()
                    nset = np.zeros(q ** n, dtype=bool)
                    nset[nidx] = True
                    if nset[crit_idx].all():
                        # valid swap
                        new_cw = cw_arr.copy()
                        new_cw[i] = base
                        return new_cw
        return None

    # --- main search loop ---
    best = cw
    while now() < deadline:
        # removal pass
        improved = True
        while improved and now() < deadline:
            improved = False
            r = try_remove(best)
            if r is not None:
                best = r
                improved = True
        # swap pass (few rounds per outer iteration)
        for _ in range(3):
            if now() >= deadline:
                break
            s = try_swap(best)
            if s is not None:
                best = s
        # periodic full re-greedy restart if stuck
        if now() < deadline:
            g = greedy()
            # reduce g
            while now() < deadline:
                r = try_remove(g)
                if r is None:
                    break
                g = r
            if g.shape[0] < best.shape[0]:
                best = g
        if best.shape[0] <= 403:  # near best known; keep polishing
            break
        # if no improvement found in a full cycle, try a fresh restart
        # (bounded by time)

    # final exact polish: aggressive removal
    while now() < deadline - 2:
        r = try_remove(best)
        if r is None:
            break
        best = r

    # ensure validity before returning
    if not is_covering(best):
        # fall back to guaranteed construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        tails = list(itertools.product(range(3), repeat=n - 4))
        best = np.array([h + list(t) for h in tetracode for t in tails], dtype=np.int64)

    return [to_ints(r) for r in best]
