"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit


def construct():
    t0 = time.perf_counter()

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Deterministic baseline: all norm^2 = 2 vectors
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    # ------------------------------------------------------------------
    # 2) Generate all vectors with norm^2 = m
    # ------------------------------------------------------------------
    def gen_all_vectors_m(m):
        if m < 1:
            return []
        results = []
        if m == 2:
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        elif m == 3:
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
        elif m == 4:
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                        results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        elif m == 5:
            for combo in itertools.combinations(range(D), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (2, -2):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
        return results

    # ------------------------------------------------------------------
    # 3) Local search: start from a valid set, try to add more vectors
    #    by exploring the full candidate pool with a smarter strategy
    # ------------------------------------------------------------------
    def try_add_vectors(initial_set, all_candidates, time_limit=60.0):
        """
        Start with initial_set (list of vector indices into all_candidates),
        try to add more vectors using a priority-based greedy that
        repeatedly picks the best candidate that can be added.
        """
        n_cand = len(all_candidates)
        if n_cand == 0:
            return list(initial_set)

        V = np.array(all_candidates, dtype=np.int64)
        norms_sq = np.array([np.dot(v, v) for v in all_candidates], dtype=np.int64)

        # Precompute Gram matrix
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)

        # For pair (i,j), the constraint is:
        # dist^2(i,j) >= max(norm^2(i), norm^2(j))
        # norm^2(i) + norm^2(j) - 2*G[i,j] >= max(norm^2(i), norm^2(j))
        # min(norm^2(i), norm^2(j)) - 2*G[i,j] >= 0
        # G[i,j] <= min(norm^2(i), norm^2(j)) // 2

        # Precompute allowed IP for each pair
        max_norm_sq = np.max(norms_sq)
        min_norm_sq = np.min(norms_sq)

        # For a set S, a candidate c can be added if for all s in S:
        # G[c,s] <= min(norm^2(c), norm^2(s)) // 2

        chosen = set(initial_set)
        chosen_list = list(chosen)

        # Track for each candidate the max violation with current chosen set
        # violation[c] = max over s in chosen of (G[c,s] - min(norm^2(c), norm^2(s))//2)
        # Can add c if violation[c] <= 0

        def compute_violations(chosen_set):
            if not chosen_set:
                return np.zeros(n_cand, dtype=np.int64)
            chosen_arr = np.array(list(chosen_set), dtype=np.int64)
            # G[:, chosen_arr] gives (n_cand, len(chosen))
            G_sub = G[:, chosen_arr]
            # norms for chosen
            norms_chosen = norms_sq[chosen_arr]
            # min(norm^2(c), norm^2(s)) for each pair
            # shape: (n_cand, len(chosen))
            min_norms = np.minimum(norms_sq[:, None], norms_chosen[None, :])
            allowed = min_norms // 2
            violations = G_sub - allowed
            return np.max(violations, axis=1)

        if not chosen:
            viol = np.zeros(n_cand, dtype=np.int64)
        else:
            viol = compute_violations(chosen)

        t_start = elapsed()
        added = 0

        # Iteratively add the best candidate
        while not is_deadline() and (elapsed() - t_start) < time_limit:
            # Find candidates with violation <= 0
            valid_mask = viol <= 0
            if not np.any(valid_mask):
                break

            # Among valid candidates, prefer those with smaller norm (easier to fit)
            # or those that are "least conflicting" with others
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                break

            # Pick the one with smallest norm (to keep max_norm low)
            # or the one that leaves the most room for others
            # Simple heuristic: smallest norm first
            valid_norms = norms_sq[valid_indices]
            best_local = valid_indices[np.argmin(valid_norms)]

            # Add it
            chosen.add(best_local)
            chosen_list.append(best_local)
            added += 1

            # Update violations for all non-chosen candidates
            # For each k not in chosen: viol[k] = max(viol[k], G[k,best_local] - min(norm^2(k), norm^2(best_local))//2)
            nb_norm = norms_sq[best_local]
            new_viol = G[:, best_local] - (np.minimum(norms_sq, nb_norm) // 2)
            viol = np.maximum(viol, new_viol)
            viol[list(chosen)] = -999999

        return chosen_list

    # ------------------------------------------------------------------
    # 4) Try multiple initial sets and candidate pools
    # ------------------------------------------------------------------
    # Strategy A: Start with base (norm^2=2), try to add from m=3,4,5
    if not is_deadline():
        all_vecs = []
        for m in [2, 3, 4, 5]:
            vecs = gen_all_vectors_m(m)
            for v in vecs:
                all_vecs.append(v)

        if len(all_vecs) > 0:
            V_all = np.array(all_vecs, dtype=np.int64)
            norms_all = np.array([np.dot(v, v) for v in all_vecs], dtype=np.int64)

            # Initial set: all norm^2=2 vectors (the base)
            init_mask = norms_all == 2
            init_indices = list(np.where(init_mask)[0])

            if len(init_indices) > 0:
                result = try_add_vectors(init_indices, all_vecs, time_limit=45.0)
                if len(result) > best_k:
                    best = V_all[result]
                    best_k = len(result)

    # ------------------------------------------------------------------
    # 5) Strategy B: Greedy from scratch with multiple random orderings
    #    on the combined pool of m=2,3,4
    # ------------------------------------------------------------------
    if not is_deadline():
        all_vecs_b = []
        for m in [2, 3, 4]:
            vecs = gen_all_vectors_m(m)
            for v in vecs:
                all_vecs_b.append(v)

        if len(all_vecs_b) > 0:
            V_b = np.array(all_vecs_b, dtype=np.int64)
            n_b = len(all_vecs_b)
            norms_b = np.array([np.dot(v, v) for v in all_vecs_b], dtype=np.int64)

            G_b = V_b.astype(np.int64) @ V_b.astype(np.int64).T
            G_b = np.ascontiguousarray(G_b)

            # Precompute allowed IP matrix
            # allowed[i,j] = min(norm^2(i), norm^2(j)) // 2
            allowed_b = np.minimum(norms_b[:, None], norms_b[None, :]) // 2

            best_b = []
            best_b_len = 0

            rng = np.random.RandomState(123)
            n_trials = 15

            for trial in range(n_trials):
                if is_deadline():
                    break

                if trial == 0:
                    order = list(range(n_b))
                elif trial == 1:
                    order = list(range(n_b - 1, -1, -1))
                else:
                    order = rng.permutation(n_b).tolist()

                chosen_mask = np.zeros(n_b, dtype=bool)
                # Track max violation with chosen for each candidate
                max_viol = np.full(n_b, -999999, dtype=np.int64)
                chosen = []

                for idx in order:
                    if is_deadline():
                        break
                    if chosen_mask[idx]:
                        continue
                    if max_viol[idx] <= 0:
                        chosen.append(idx)
                        chosen_mask[idx] = True
                        # Update max_viol for all others
                        new_viol = G_b[:, idx] - allowed_b[:, idx]
                        max_viol = np.maximum(max_viol, new_viol)
                        max_viol[chosen_mask] = -999999

                if len(chosen) > best_b_len:
                    best_b_len = len(chosen)
                    best_b = chosen

            if best_b and best_b_len > best_k:
                best = V_b[best_b]
                best_k = best_b_len

    # ------------------------------------------------------------------
    # 6) Strategy C: Local search / hill climbing on the selected set
    #    Try to swap out one vector for one or more that fit better
    # ------------------------------------------------------------------
    if not is_deadline() and best_k > 0:
        # Use the current best set and try to improve via swaps
        # This is a simplified version: try removing one vector and
        # greedily adding as many as possible from the remaining pool

        # Build a candidate pool from m=2,3,4
        all_vecs_c = []
        for m in [2, 3, 4]:
            vecs = gen_all_vectors_m(m)
            for v in vecs:
                all_vecs_c.append(v)

        if len(all_vecs_c) > 0:
            V_c = np.array(all_vecs_c, dtype=np.int64)
            n_c = len(all_vecs_c)
            norms_c = np.array([np.dot(v, v) for v in all_vecs_c], dtype=np.int64)

            G_c = V_c.astype(np.int64) @ V_c.astype(np.int64).T
            G_c = np.ascontiguousarray(G_c)
            allowed_c = np.minimum(norms_c[:, None], norms_c[None, :]) // 2

            # Current best set
            current_best = best.astype(np.int64)

            # Find indices in V_c that match the current best (approximate)
            # Since we generated the same vectors, we can try to match
            # For simplicity, just run a fresh greedy with the current best as seed

            # Try: start with a subset of the current best, then add greedily
            # Actually, let's just try a few more random orderings with a
            # slightly different strategy: prioritize vectors with norm^2=2
            # early, then norm^2=3, etc.

            rng2 = np.random.RandomState(456)
            n_trials_c = 10

            for trial in range(n_trials_c):
                if is_deadline():
                    break

                if trial == 0:
                    # Sort by norm, then by index
                    order = list(np.argsort(norms_c))
                else:
                    order = rng2.permutation(n_c).tolist()

                chosen_mask = np.zeros(n_c, dtype=bool)
                max_viol = np.full(n_c, -999999, dtype=np.int64)
                chosen = []

                for idx in order:
                    if is_deadline():
                        break
                    if chosen_mask[idx]:
                        continue
                    if max_viol[idx] <= 0:
                        chosen.append(idx)
                        chosen_mask[idx] = True
                        new_viol = G_c[:, idx] - allowed_c[:, idx]
                        max_viol = np.maximum(max_viol, new_viol)
                        max_viol[chosen_mask] = -999999

                if len(chosen) > best_k:
                    best = V_c[chosen]
                    best_k = len(chosen)

    # ------------------------------------------------------------------
    # 7) Final safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
