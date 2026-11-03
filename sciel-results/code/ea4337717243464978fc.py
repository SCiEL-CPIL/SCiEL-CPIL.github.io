import time
import random
import numpy as np
from itertools import product

_DEADLINE = 145.0  # Safety margin vs 150s wall-clock kill


def _all_words(q, n):
    return np.array(list(product(range(q), repeat=n)), dtype=np.int64)


def _build_neighbors(q, n, radius):
    total = q ** n
    base = _all_words(q, n)
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    idx_of = (base * powers).sum(axis=1)

    if radius == 1:
        ball_size = 1 + n * (q - 1)
        balls = np.zeros((total, ball_size), dtype=np.int64)
        balls[:, 0] = idx_of
        k = 1
        for pos in range(n):
            for delta in range(1, q):
                flipped = base.copy()
                flipped[:, pos] = (base[:, pos] + delta) % q
                balls[:, k] = (flipped * powers).sum(axis=1)
                k += 1
    else:
        # Generic BFS for small spaces
        balls_list = []
        for i in range(total):
            seen = {i}
            frontier = [i]
            for _ in range(radius):
                nxt = []
                for widx in frontier:
                    w = base[widx]
                    for pos in range(n):
                        for v in range(q):
                            if v == w[pos]:
                                continue
                            nw = w.copy()
                            nw[pos] = v
                            j = int((nw * powers).sum())
                            if j not in seen:
                                seen.add(j)
                                nxt.append(j)
                frontier = nxt
            balls_list.append(np.array(sorted(seen), dtype=np.int64))
        maxb = max(len(b) for b in balls_list)
        balls = np.zeros((total, maxb), dtype=np.int64)
        for i, b in enumerate(balls_list):
            balls[i, :len(b)] = b

    # Reverse neighborhood: reverse[i] = list of codewords whose ball contains i
    rev_counts = np.zeros(total, dtype=np.int64)
    for i in range(total):
        rev_counts[balls[i]] += 1
    max_rev = int(rev_counts.max())
    reverse = np.zeros((total, max_rev), dtype=np.int64)
    fill = np.zeros(total, dtype=np.int64)
    for i in range(total):
        for j in balls[i]:
            reverse[j, fill[j]] = i
            fill[j] += 1

    return balls, reverse, rev_counts, base, idx_of


def _greedy_cover(balls, total, budget_time, rng):
    """Greedy set cover with vectorized gain computation."""
    cov = np.zeros(total, dtype=np.int32)
    chosen = []
    in_code = np.zeros(total, dtype=bool)

    t0 = time.time()
    while True:
        if time.time() - t0 > budget_time:
            break
        uncovered = (cov == 0)
        if not uncovered.any():
            break

        # Vectorized gain: number of uncovered words each candidate covers
        gains = uncovered[balls].sum(axis=1)
        max_gain = int(gains.max())
        if max_gain == 0:
            break
        top = np.where(gains == max_gain)[0]
        best = int(rng.choice(top))

        chosen.append(best)
        in_code[best] = True
        cov[balls[best]] += 1

    return chosen, in_code, cov


def _reduce_fast(balls, total, chosen):
    """Linear pass: drop any codeword that is fully redundant."""
    if not chosen:
        return list(chosen)
    counts = np.zeros(total, dtype=np.int16)
    for c in chosen:
        counts[balls[c]] += 1
    new_code = []
    for c in chosen:
        counts[balls[c]] -= 1
        if (counts == 0).any():
            counts[balls[c]] += 1
            new_code.append(c)
    return new_code


def _local_search(balls, reverse, rev_counts, total, chosen, in_code, covcnt, budget_time, rng):
    """Local search: redundancy removal + exact 1-for-1 swap via reverse-neighborhood intersection."""
    improved = True
    t0 = time.time()
    rev_list = [list(reverse[i, :rev_counts[i]]) for i in range(total)]

    while improved and time.time() - t0 < budget_time:
        improved = False

        # Pass A: Redundancy Removal
        for idx in range(len(chosen)):
            if time.time() - t0 > budget_time:
                break
            i = chosen[idx]
            if (covcnt[balls[i]] >= 2).all():
                covcnt[balls[i]] -= 1
                in_code[i] = False
                chosen.pop(idx)
                improved = True
                break

        if improved:
            continue

        # Pass B: 1-for-1 Swap using reverse-neighborhood intersection
        if time.time() - t0 < budget_time - 1.0:
            n_chosen = len(chosen)
            order = list(range(n_chosen))
            rng.shuffle(order)

            for idx in order:
                if time.time() - t0 > budget_time:
                    break
                i = chosen[idx]
                b_i = balls[i]

                # Tentatively remove
                covcnt[b_i] -= 1
                critical = b_i[covcnt[b_i] == 1]
                if critical.size == 0:
                    # Fully redundant
                    in_code[i] = False
                    chosen.pop(idx)
                    improved = True
                    break

                # Intersect reverse neighborhoods of critical words
                # A candidate j is valid iff j is in reverse[cw] for every critical cw
                # and j is not already in the code.
                first = critical[0]
                candidates = [j for j in rev_list[first] if not in_code[j]]
                for cw in critical[1:]:
                    if not candidates:
                        break
                    rc = rev_list[cw]
                    rc_set = set(rc)
                    candidates = [j for j in candidates if j in rc_set]

                if candidates:
                    j = rng.choice(candidates)
                    # commit: remove i, add j
                    covcnt[b_i] += 1  # revert tentative removal
                    in_code[i] = False
                    chosen.pop(idx)
                    covcnt[balls[j]] += 1
                    in_code[j] = True
                    chosen.append(j)
                    improved = True
                    break
                else:
                    # revert
                    covcnt[b_i] += 1

    return chosen, in_code, covcnt


def _repair(balls, total, chosen, in_code, covcnt, budget_time):
    """Repair: add codewords to cover any uncovered words."""
    t0 = time.time()
    covered = np.zeros(total, dtype=bool)
    for i in chosen:
        covered[balls[i]] = True

    while time.time() - t0 < budget_time:
        uncovered = ~covered
        if not uncovered.any():
            break

        gains = uncovered[balls].sum(axis=1)
        mask = ~in_code
        gains = np.where(mask, gains, -1)
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break

        chosen.append(best)
        in_code[best] = True
        covcnt[balls[best]] += 1
        covered[balls[best]] = True

    return chosen, in_code, covcnt


def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    rng = random.Random(42)

    # Fallback for very large spaces
    if total > 5_000_000:
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tet for tail in product(range(q), repeat=n - 4)]
        return [list(range(n))]

    balls, reverse, rev_counts, base, idx_of = _build_neighbors(q, n, radius)

    def time_left():
        return _DEADLINE - (time.time() - t0)

    best_chosen = None
    best_len = total + 1

    # Phase 1: Greedy initialization with multiple random restarts
    num_restarts = 6
    restart_budget = max(1.5, time_left() / (num_restarts * 3))

    for restart in range(num_restarts):
        if time_left() < 8.0:
            break

        cov = np.zeros(total, dtype=np.int32)
        chosen = []
        in_code = np.zeros(total, dtype=bool)

        greedy_t0 = time.time()
        greedy_budget = min(restart_budget, time_left() - 4.0)

        while time.time() - greedy_t0 < greedy_budget:
            uncovered = (cov == 0)
            if not uncovered.any():
                break
            gains = uncovered[balls].sum(axis=1)
            max_gain = int(gains.max())
            if max_gain == 0:
                break
            top = np.where(gains == max_gain)[0]
            best = int(rng.choice(top))
            chosen.append(best)
            in_code[best] = True
            cov[balls[best]] += 1

        # Fast reduce
        chosen = _reduce_fast(balls, total, chosen)

        # Rebuild state
        in_code = np.zeros(total, dtype=bool)
        for i in chosen:
            in_code[i] = True
        covcnt = np.zeros(total, dtype=np.int32)
        for i in chosen:
            covcnt[balls[i]] += 1

        # Local search
        ls_budget = min(12.0, time_left() - 4.0)
        chosen, in_code, covcnt = _local_search(balls, reverse, rev_counts, total, chosen, in_code, covcnt, ls_budget, rng)

        # Repair if needed
        chosen, in_code, covcnt = _repair(balls, total, chosen, in_code, covcnt, max(1.0, time_left() - 2.0))

        if len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = list(chosen)

    # Phase 2: Iterated Local Search (Perturb and Repair)
    if best_chosen is not None and time_left() > 15.0:
        ils_budget = time_left() - 8.0
        ils_t0 = time.time()

        current_chosen = list(best_chosen)
        current_len = len(current_chosen)
        current_in_code = np.zeros(total, dtype=bool)
        for i in current_chosen:
            current_in_code[i] = True
        current_covcnt = np.zeros(total, dtype=np.int32)
        for i in current_chosen:
            current_covcnt[balls[i]] += 1

        while time.time() - ils_t0 < ils_budget:
            # Perturb: remove a few random codewords
            perturb_size = min(3, max(2, len(current_chosen) // 60))
            if len(current_chosen) <= perturb_size:
                break

            to_remove = rng.sample(current_chosen, perturb_size)
            for i in to_remove:
                current_covcnt[balls[i]] -= 1
                current_in_code[i] = False
                current_chosen.remove(i)

            # Repair
            repair_budget = min(4.0, time_left() - 4.0)
            current_chosen, current_in_code, current_covcnt = _repair(
                balls, total, current_chosen, current_in_code, current_covcnt, repair_budget
            )

            # Local search
            ls_budget = min(8.0, time_left() - 4.0)
            current_chosen, current_in_code, current_covcnt = _local_search(
                balls, reverse, rev_counts, total, current_chosen, current_in_code, current_covcnt, ls_budget, rng
            )

            if len(current_chosen) < best_len:
                best_len = len(current_chosen)
                best_chosen = list(current_chosen)

    # Final verification and repair
    if best_chosen is not None and time_left() > 2.0:
        covered = np.zeros(total, dtype=bool)
        for i in best_chosen:
            covered[balls[i]] = True

        if not covered.all():
            in_code = np.zeros(total, dtype=bool)
            for i in best_chosen:
                in_code[i] = True
            covcnt = np.zeros(total, dtype=np.int32)
            for i in best_chosen:
                covcnt[balls[i]] += 1

            best_chosen, in_code, covcnt = _repair(balls, total, best_chosen, in_code, covcnt, max(1.0, time_left() - 0.5))

    if best_chosen is None:
        best_chosen, _, _ = _greedy_cover(balls, total, max(1.0, time_left()), rng)

    return [base[i].tolist() for i in best_chosen]
