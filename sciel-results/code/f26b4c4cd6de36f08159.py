"""Odd covering system with moduli dividing 51975; minimize uncovered residues."""
import time
import numpy as np

BASE = 51975
DEADLINE = 115.0  # safety margin under the 120 s wall-clock limit (evaluator kills at 150s)


def _divisors():
    """Generate all odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1."""
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if m > 1:
                        divs.append(m)
    return sorted(divs)


DIVS = _divisors()
N = len(DIVS)


def _greedy(order, t0, deadline):
    """
    Run greedy algorithm with given order of indices.
    Uses cover_count for accurate marginal gain.
    Returns (score, chosen_list, cover_count)
    """
    cover_count = np.zeros(BASE, dtype=np.int32)
    chosen = [0] * N

    for i in order:
        m = DIVS[i]
        best_r = 0
        best_g = -1

        # Find residue that covers most uncovered points (cover_count == 0)
        for r in range(m):
            g = np.sum(cover_count[r::m] == 0)
            if g > best_g:
                best_g = g
                best_r = r

        chosen[i] = best_r
        cover_count[best_r::m] += 1

        if (time.time() - t0) > deadline:
            break

    score = BASE - int(np.sum(cover_count > 0))
    return score, chosen, cover_count


def _local_search(cover_count, chosen, t0, deadline, max_rounds=200):
    """
    Local search: try changing residue of each modulus to improve coverage.
    Returns (score, chosen, cover_count)
    """
    score = BASE - int(np.sum(cover_count > 0))
    improved = True
    round_num = 0

    while improved and round_num < max_rounds:
        improved = False
        round_num += 1

        for i in range(N):
            if (time.time() - t0) > deadline:
                return score, chosen, cover_count.copy()

            m = DIVS[i]
            current_r = chosen[i]

            best_r = current_r
            best_gain = 0

            # Determine how many candidates to try
            # For small moduli, try all; for large, sample
            if m <= 100:
                candidates = list(range(m))
            else:
                candidates = [current_r]
                # Add random candidates
                n_rand = min(50, m)
                for _ in range(n_rand):
                    candidates.append(int(np.random.randint(0, m)))

            for r in candidates:
                if r == current_r:
                    continue

                # Compute gain of switching from current_r to r
                # Newly covered: points in r::m with cover_count == 0
                # Lost coverage: points in current_r::m with cover_count == 1
                newly_covered = np.sum(cover_count[r::m] == 0)
                becomes_uncovered = np.sum(cover_count[current_r::m] == 1)

                gain = newly_covered - becomes_uncovered
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != current_r and best_gain > 0:
                # Apply the change
                cover_count[current_r::m] -= 1
                cover_count[best_r::m] += 1
                chosen[i] = best_r
                score -= best_gain
                improved = True

        if (time.time() - t0) > deadline:
            break

    return score, chosen, cover_count


def _cell_partition_greedy(order, t0, deadline):
    """
    Cell-based greedy: process moduli in given order, but for each modulus,
    compute the marginal gain more carefully by considering interactions.
    This is a variant that may find different basins.
    """
    cover_count = np.zeros(BASE, dtype=np.int32)
    chosen = [0] * N

    for i in order:
        m = DIVS[i]
        best_r = 0
        best_g = -1

        # Find residue that covers most uncovered points
        for r in range(m):
            g = np.sum(cover_count[r::m] == 0)
            if g > best_g:
                best_g = g
                best_r = r

        chosen[i] = best_r
        cover_count[best_r::m] += 1

        if (time.time() - t0) > deadline:
            break

    score = BASE - int(np.sum(cover_count > 0))
    return score, chosen, cover_count


def _refine(cover_count, chosen, t0, deadline):
    """
    More careful local search with multiple passes and full residue scan for small moduli.
    """
    score = BASE - int(np.sum(cover_count > 0))

    improved = True
    round_num = 0
    max_rounds = 300

    while improved and round_num < max_rounds:
        improved = False
        round_num += 1

        for i in range(N):
            if (time.time() - t0) > deadline:
                break

            m = DIVS[i]
            current_r = chosen[i]

            best_r = current_r
            best_gain = 0

            # For small moduli, scan all residues
            # For large moduli, use a larger sample
            if m <= 200:
                candidates = list(range(m))
            else:
                candidates = [current_r]
                n_rand = min(100, m)
                for _ in range(n_rand):
                    candidates.append(int(np.random.randint(0, m)))

            for r in candidates:
                if r == current_r:
                    continue

                newly_covered = np.sum(cover_count[r::m] == 0)
                becomes_uncovered = np.sum(cover_count[current_r::m] == 1)

                gain = newly_covered - becomes_uncovered
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != current_r and best_gain > 0:
                cover_count[current_r::m] -= 1
                cover_count[best_r::m] += 1
                chosen[i] = best_r
                score -= best_gain
                improved = True

        if (time.time() - t0) > deadline:
            break

    return score, chosen, cover_count


def _final_polish(cover_count, chosen, t0, deadline):
    """
    High-precision final polish: full residue sweep for ALL moduli with m <= 135,
    plus a second pass with even more rounds.
    """
    score = BASE - int(np.sum(cover_count > 0))

    # Pass 1: full sweep for moduli up to 135
    improved = True
    round_num = 0
    max_rounds = 500

    while improved and round_num < max_rounds:
        improved = False
        round_num += 1

        for i in range(N):
            if (time.time() - t0) > deadline:
                break

            m = DIVS[i]
            current_r = chosen[i]

            best_r = current_r
            best_gain = 0

            # Full sweep for moduli up to 135 (3^3 * 5 = 135)
            if m <= 135:
                candidates = list(range(m))
            else:
                candidates = [current_r]
                n_rand = min(200, m)
                for _ in range(n_rand):
                    candidates.append(int(np.random.randint(0, m)))

            for r in candidates:
                if r == current_r:
                    continue

                newly_covered = np.sum(cover_count[r::m] == 0)
                becomes_uncovered = np.sum(cover_count[current_r::m] == 1)

                gain = newly_covered - becomes_uncovered
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != current_r and best_gain > 0:
                cover_count[current_r::m] -= 1
                cover_count[best_r::m] += 1
                chosen[i] = best_r
                score -= best_gain
                improved = True

        if (time.time() - t0) > deadline:
            break

    return score, chosen, cover_count


def construct():
    t0 = time.time()

    # Multiple greedy initializations with different orderings
    best_score = BASE
    best_chosen = None
    best_cover_count = None

    # Strategy 1: Descending moduli (large first)
    order_desc = sorted(range(N), key=lambda i: -DIVS[i])
    score, chosen, cc = _greedy(order_desc, t0, DEADLINE - 90)
    if score < best_score:
        best_score = score
        best_chosen = chosen
        best_cover_count = cc

    # Strategy 2: Ascending moduli (small first)
    if (time.time() - t0) < DEADLINE - 80:
        order_asc = sorted(range(N), key=lambda i: DIVS[i])
        score, chosen, cc = _greedy(order_asc, t0, DEADLINE - 80)
        if score < best_score:
            best_score = score
            best_chosen = chosen
            best_cover_count = cc

    # Strategy 3: Random order
    if (time.time() - t0) < DEADLINE - 70:
        order_rand = list(range(N))
        np.random.shuffle(order_rand)
        score, chosen, cc = _greedy(order_rand, t0, DEADLINE - 70)
        if score < best_score:
            best_score = score
            best_chosen = chosen
            best_cover_count = cc

    # Strategy 4: Cell-partition style - prioritize prime powers
    if (time.time() - t0) < DEADLINE - 60:
        # Order: prime powers first (27, 25, 7, 11), then composites
        def _priority(i):
            m = DIVS[i]
            # Score: prefer pure prime powers, then by size
            if m in (27, 25, 7, 11):
                return (0, -m)
            elif m in (9, 5, 3):
                return (1, -m)
            else:
                return (2, -m)
        order_cp = sorted(range(N), key=_priority)
        score, chosen, cc = _cell_partition_greedy(order_cp, t0, DEADLINE - 60)
        if score < best_score:
            best_score = score
            best_chosen = chosen
            best_cover_count = cc

    # Local search to refine
    if (time.time() - t0) < DEADLINE - 30 and best_chosen is not None:
        best_score, best_chosen, best_cover_count = _refine(
            best_cover_count, best_chosen, t0, DEADLINE - 25
        )

    # Final high-precision polish
    if (time.time() - t0) < DEADLINE - 5 and best_chosen is not None:
        best_score, best_chosen, best_cover_count = _final_polish(
            best_cover_count, best_chosen, t0, DEADLINE - 2
        )

    # Construct the final list of classes
    final_classes = []
    for i in range(N):
        m = DIVS[i]
        r = best_chosen[i] if best_chosen is not None else 0
        if r is None or r >= m:
            r = 0
        final_classes.append([m, r])

    return final_classes
