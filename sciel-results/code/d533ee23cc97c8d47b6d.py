"""Greedy + randomized-restart + swap-polish covering code for K_q(n, radius)."""
import time
import numpy as np


def _build_tables(q, n, radius):
    # Precompute the "sphere" of each word (all words within Hamming distance radius).
    # Represent the whole space as an array of shape (q**n, n) of ints.
    space = np.array(
        [list(map(int, np.base_repr(i, base=q).zfill(n))) for i in range(q ** n)],
        dtype=np.int8,
    )
    # sphere[word] = boolean mask (length q**n) of words within distance radius.
    sphere = np.zeros((q ** n, q ** n), dtype=bool)
    for w in range(q ** n):
        diffs = np.abs(space - space[w])
        d = (diffs != 0).sum(axis=1)
        sphere[w] = d <= radius
    return space, sphere


def _greedy_cover(sphere, max_iters):
    """Greedy set cover: repeatedly pick the word covering the most uncovered words."""
    m = sphere.shape[0]
    uncovered = np.ones(m, dtype=bool)
    chosen = []
    for _ in range(max_iters):
        if not uncovered.any():
            break
        # count how many uncovered words each candidate covers
        counts = sphere[:, uncovered].sum(axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        chosen.append(best)
        uncovered[sphere[best]] = False
    return chosen, uncovered


def _swap_polish(sphere, chosen, max_rounds):
    """Try to replace a chosen word with one that covers more uncovered words."""
    m = sphere.shape[0]
    chosen_set = set(chosen)
    covered = np.zeros(m, dtype=bool)
    for c in chosen:
        covered |= sphere[c]
    uncovered = ~covered
    for _ in range(max_rounds):
        if not uncovered.any():
            break
        improved = False
        # for each chosen word, find a replacement that covers more uncovered words
        for idx, c in enumerate(chosen):
            # uncovered words that c covers
            c_uncovered = sphere[c] & uncovered
            if not c_uncovered.any():
                continue
            # candidate replacement: word covering the most of c_uncovered (and possibly more)
            cand_counts = sphere[:, c_uncovered].sum(axis=1)
            cand_counts[chosen_set] = -1  # don't pick an already-chosen word
            best = int(np.argmax(cand_counts))
            if cand_counts[best] > cand_counts[c]:
                # swapping c -> best reduces uncovered
                chosen[idx] = best
                chosen_set = set(chosen)
                improved = True
                # recompute covered/uncovered
                covered = np.zeros(m, dtype=bool)
                for cc in chosen:
                    covered |= sphere[cc]
                uncovered = ~covered
                break
        if not improved:
            break
    return chosen


def construct(q, n, radius):
    start = time.time()
    deadline = start + 120.0  # safety margin under 150 s

    space, sphere = _build_tables(q, n, radius)
    m = sphere.shape[0]

    # baseline: all words
    best = list(range(m))
    best_len = m

    # deterministic greedy first
    g, unc = _greedy_cover(sphere, m)
    g = _swap_polish(sphere, list(g), 5)
    if len(g) < best_len:
        best, best_len = g, len(g)

    # randomized restarts within budget
    rng = np.random.default_rng(12345)
    restarts = 0
    while time.time() < deadline - 25.0 and restarts < 60:
        restarts += 1
        # randomize by permuting candidate order via a random tiebreak noise
        noise = rng.random(m) * 1e-6
        # We emulate randomization by starting from a random subset seed
        # then greedy-completing.
        seed_size = max(1, m // 200)
        seed_idx = rng.choice(m, size=seed_size, replace=False)
        chosen = [int(i) for i in seed_idx]
        covered = np.zeros(m, dtype=bool)
        for c in chosen:
            covered |= sphere[c]
        uncovered = ~covered
        # greedy complete
        for _ in range(m):
            if not uncovered.any():
                break
            counts = sphere[:, uncovered].sum(axis=1)
            counts += noise
            bestw = int(np.argmax(counts))
            if sphere[bestw] & uncovered:
                if not covered[bestw].any():  # avoid duplicates
                    chosen.append(bestw)
                    covered |= sphere[bestw]
                    uncovered = ~covered
        # polish
        chosen = _swap_polish(sphere, chosen, 4)
        if len(chosen) < best_len:
            best, best_len = chosen, len(chosen)

    # final polish on the best if time allows
    if time.time() < deadline - 10.0:
        best = _swap_polish(sphere, list(best), 10)

    # return as list of lists of ints
    return [space[w].tolist() for w in best]
