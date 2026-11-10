# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _balls(q, n, radius):
    """Return list of lists; balls[c] = set of word-indices within Hamming distance radius of codeword c.
    Words are indexed 0..q^n-1 in mixed-radix (base q, little-endian)."""
    total = q ** n
    # precompute all words as arrays
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)  # total x n
    # For each word, its ball = itself + all single-coordinate changes (radius 1). For general radius,
    # we only support radius 1 efficiently; fallback brute force for other radii.
    if radius == 1:
        balls = [None] * total
        for c in range(total):
            w = words[c]
            covered = {c}
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    # index of word with pos changed to val
                    idx = c + (val - orig) * (q ** pos)
                    covered.add(idx)
            balls[c] = covered
        return words, balls
    else:
        # brute force
        words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        idx_map = {}
        for i in range(total):
            idx_map[tuple(words[i])] = i
        balls = [None] * total
        for c in range(total):
            w = words[c]
            covered = set()
            for cand in range(total):
                d = int(np.sum(words[cand] != w))
                if d <= radius:
                    covered.add(cand)
            balls[c] = covered
        return words, balls


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    total = q ** n
    words, balls = _balls(q, n, radius)
    ball_list = balls  # list of sets
    ball_arr = [np.fromiter(b, dtype=np.int32) for b in ball_list]

    # Greedy set cover: repeatedly pick codeword covering most uncovered words.
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    chosen = []
    # We'll maintain for each candidate the count of uncovered words it covers.
    # Greedy: at each step pick argmax. To be fast, recompute counts by scanning candidates
    # against uncovered mask using numpy.
    # Precompute each ball as boolean mask for speed of counting.
    ball_masks = [np.zeros(total, dtype=bool) for _ in range(total)]
    for c in range(total):
        ball_masks[c][np.fromiter(ball_list[c], dtype=np.int32)] = True

    while uncovered_count > 0 and time.time() < deadline:
        # counts = number of uncovered words in each ball
        counts = np.array([int(np.sum(ball_masks[c][uncovered])) for c in range(total)], dtype=np.int32)
        c = int(np.argmax(counts))
        if counts[c] == 0:
            break
        chosen.append(c)
        uncovered[ball_masks[c]] = False
        uncovered_count -= int(counts[c])

    chosen_set = set(chosen)
    chosen = sorted(chosen_set)

    def is_covered_by(chosen_list):
        # returns True if every word is within radius of some chosen
        if not chosen_list:
            return total == 0
        covered = np.zeros(total, dtype=bool)
        for c in chosen_list:
            covered |= ball_masks[c]
        return bool(np.all(covered))

    # Local search: try removing codewords, then swapping one out for another.
    # Represent chosen as list; maintain covered mask.
    def build_covered(chosen_list):
        cov = np.zeros(total, dtype=bool)
        for c in chosen_list:
            cov |= ball_masks[c]
        return cov

    chosen = list(chosen)
    covered = build_covered(chosen)
    assert is_covered_by(chosen)

    # Phase 1: greedy removal (try removing each; if still covered, remove)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_chosen = []
        for c in chosen:
            # temporarily remove c
            new_cov = covered & ~ball_masks[c]
            if bool(np.all(new_cov)):
                covered = new_cov
                improved = True
            else:
                new_chosen.append(c)
        chosen = new_chosen

    # Phase 2: swap moves - for each chosen c, try replacing with some d not chosen
    # such that coverage stays full and (optionally) we keep same size (swap) — but to reduce
    # size we rely on removal. Swaps can enable further removals. Do a few passes.
    rng = random.Random(12345)
    all_candidates = list(range(total))
    for _pass in range(4):
        if time.time() >= deadline:
            break
        swapped = False
        # random order
        order = chosen[:]
        rng.shuffle(order)
        for c in order:
            if time.time() >= deadline:
                break
            # try swapping c -> d for a random subset of d
            d_candidates = [d for d in all_candidates if d not in chosen_set]
            rng.shuffle(d_candidates)
            # limit trials per c
            for d in d_candidates[:80]:
                new_cov = (covered | ball_masks[d]) & ~ball_masks[c]
                if bool(np.all(new_cov)):
                    covered = new_cov
                    chosen = [x for x in chosen if x != c]
                    chosen.append(d)
                    chosen_set = set(chosen)
                    swapped = True
                    # after a swap, try to remove again
                    rem = True
                    while rem and time.time() < deadline:
                        rem = False
                        new_chosen = []
                        for cc in chosen:
                            nc = covered & ~ball_masks[cc]
                            if bool(np.all(nc)):
                                covered = nc
                                rem = True
                            else:
                                new_chosen.append(cc)
                        chosen = new_chosen
                        chosen_set = set(chosen)
                    break
        if not swapped:
            break

    # Final safety: ensure valid; if not, fall back to greedy (recompute)
    if not is_covered_by(chosen):
        # rebuild from greedy chosen
        chosen = sorted(chosen_set)
        if not is_covered_by(chosen):
            # ultimate fallback: all words (valid, huge)
            chosen = list(range(total))

    # Return codewords as lists of ints
    result = [words[c].tolist() for c in chosen]
    # ensure distinct
    seen = set()
    out = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(w)
    return out
# EVOLVE-BLOCK-END
