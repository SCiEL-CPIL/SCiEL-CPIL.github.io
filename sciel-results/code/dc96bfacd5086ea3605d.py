"""Greedy set-cover + prune for ternary covering code K_3(8,1), with deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _ball_memberships(q, n, radius):
    """For every space word index, list the codeword indices within radius.
    Returns list of lists."""
    total = q ** n
    # precompute offsets: for each position and each symbol change, the index delta
    # represent a word as base-q integer
    # Build all neighbor offsets (including 0)
    offsets = [0]
    for pos in range(n):
        stride = q ** pos
        for delta in range(1, q):
            offsets.append(stride * delta)
            offsets.append(-stride * delta)
    # dedupe
    offsets = list(set(offsets))
    members = [[] for _ in range(total)]
    for w in range(total):
        for off in offsets:
            v = w + off
            if 0 <= v < total:
                members[w].append(v)
    return members


def _idx_to_word(idx, q, n):
    w = []
    for _ in range(n):
        w.append(idx % q)
        idx //= q
    return w


def _word_to_idx(word, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    total = q ** n

    # Build the initial candidate pool: direct-sum of ternary tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                 for a in range(q) for b in range(q)]
    candidates = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            candidates.append(h + list(tail))

    cand_idx = [_word_to_idx(c, q, n) for c in candidates]
    cand_idx_set = set(cand_idx)

    # For each candidate, precompute the set of space-word indices in its ball
    # (space words = all q^n words; a candidate covers a space word if dist<=radius)
    # We'll compute ball membership: for each space word, which candidates cover it.
    # Instead compute per-candidate covered list.
    # radius-1 ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)

    # Build per-candidate covered space indices using offsets
    offsets = [0]
    for pos in range(n):
        stride = q ** pos
        for delta in range(1, q):
            offsets.append(stride * delta)
            offsets.append(-stride * delta)
    offsets = list(set(offsets))

    cand_covered = []
    for ci in cand_idx:
        covered = []
        for off in offsets:
            v = ci + off
            if 0 <= v < total:
                covered.append(v)
        cand_covered.append(covered)

    # uncovered mask over space words
    uncovered = np.ones(total, dtype=np.uint8)

    chosen = []
    chosen_set = set()

    # Greedy: repeatedly pick candidate covering most uncovered words.
    # To speed up, we keep a list of candidate indices and recompute coverage count
    # lazily. With ~486 candidates and 6561 words this is fine.
    cand_order = list(range(len(candidates)))

    while True:
        if time.time() > deadline - 5.0:
            break
        # find best candidate by uncovered coverage
        best = -1
        best_count = 0
        for ci in cand_order:
            cov = cand_covered[ci]
            # count uncovered in cov
            cnt = 0
            for v in cov:
                if uncovered[v]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best = ci
        if best < 0 or best_count == 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        for v in cand_covered[best]:
            uncovered[v] = 0

    # Verify full coverage (should hold)
    # Pruning: remove codewords that don't uniquely cover any word
    # Build word -> list of chosen candidates covering it
    word_coverers = [[] for _ in range(total)]
    for ci in chosen:
        for v in cand_covered[ci]:
            word_coverers[v].append(ci)

    keep = set()
    for v in range(total):
        covs = word_coverers[v]
        if len(covs) == 1:
            keep.add(covs[0])

    # Every chosen must be kept only if it uniquely covers something; but we must
    # ensure full coverage remains. Pruning a codeword that uniquely covers some word
    # is safe; removing one that never uniquely covers a word is safe.
    pruned = [ci for ci in chosen if ci in keep]

    # Safety: if pruning broke coverage (shouldn't), fall back to full chosen.
    # Re-verify coverage of pruned set.
    still_uncovered = np.ones(total, dtype=np.uint8)
    for ci in pruned:
        for v in cand_covered[ci]:
            still_uncovered[v] = 0
    if still_uncovered.any():
        pruned = chosen

    result = [candidates[ci] for ci in pruned]
    # Ensure distinct
    seen = set()
    out = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out
# EVOLVE-BLOCK-END
