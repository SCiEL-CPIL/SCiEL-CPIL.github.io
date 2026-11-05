"""Combined program: fast vectorized greedy + iterated local search (removal + swap) + polish."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words (length-4 tetracode core)."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _build_neighbors(q, n, radius):
    """Return list of neighbor index lists (words within Hamming distance <= radius)."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)
    # Map index -> word for neighbor computation
    neighbors = [None] * total
    for idx in range(total):
        w = all_words[idx].copy()
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val != orig:
                        w[pos] = val
                        nbrs.append(_word_to_idx(w, q, n))
                w[pos] = orig
        neighbors[idx] = nbrs
    return neighbors, all_words


def _verify_cover(codewords, q, n, radius, neighbors):
    """Return True if codewords cover the whole space."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = _word_to_idx(cw, q, n)
        for nbr in neighbors[idx]:
            covered[nbr] = True
    return bool(covered.all())


def _greedy_init(q, n, radius, neighbors, all_words, deadline):
    """Fast vectorized greedy: pick candidate covering most uncovered words."""
    total = q ** n
    # coverage matrix: cov[i][j] = 1 if word j is within radius of candidate i
    cov = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        for nbr in neighbors[i]:
            cov[i, nbr] = 1
    uncovered = np.ones(total, dtype=np.int8)
    code_list = []
    while True:
        if time.time() > deadline:
            break
        # uncovered count per candidate = row sum of cov where uncovered
        counts = cov.dot(uncovered)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        code_list.append(best_idx)
        uncovered[cov[best_idx] == 1] = 0
        if uncovered.sum() == 0:
            break
    return code_list


def _local_search(code_list, q, n, radius, neighbors, all_words, deadline):
    """Iterated local search: removal + swap moves. Returns best code (list of indices)."""
    total = q ** n
    code_set = set(code_list)
    # cover_count[j] = number of codewords covering word j
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    best_size = len(code_list)
    best_code = list(code_list)
    rng = random.Random(12345)

    while time.time() < deadline:
        improved = False

        # ---- Phase 1: removal pass ----
        order = list(code_list)
        rng.shuffle(order)
        for c in order:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.remove(c)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)

        if improved:
            continue

        # ---- Phase 2: swap move ----
        if len(code_list) <= 403:
            break
        c = rng.choice(code_list)
        # temporarily remove
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
        code_list.remove(c)
        code_set.discard(c)

        uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        if not uncovered_words:
            # removal was valid; keep it removed
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            continue

        # find replacement covering most uncovered_words
        unc_arr = np.zeros(total, dtype=np.int8)
        for u in uncovered_words:
            unc_arr[u] = 1
        # candidate gain = number of uncovered words it covers (excluding c)
        # recompute quickly: for each candidate, count uncovered it covers
        best_cand = -1
        best_gain = -1
        # sample candidates for speed
        sample = rng.sample(range(total), min(total, 2000))
        for cand in sample:
            if cand == c:
                continue
            gain = 0
            for nbr in neighbors[cand]:
                if unc_arr[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = cand

        if best_cand >= 0 and best_gain > 0:
            # add replacement
            code_list.append(best_cand)
            code_set.add(best_cand)
            for nbr in neighbors[best_cand]:
                cover_count[nbr] += 1
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        else:
            # restore c
            code_list.append(c)
            code_set.add(c)
            for nbr in neighbors[c]:
                cover_count[nbr] += 1

    return best_code


def _polish(code_list, q, n, radius, neighbors, all_words, deadline):
    """Final high-precision removal polish with deterministic ordering."""
    total = q ** n
    code_set = set(code_list)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    # repeated deterministic removal passes
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for c in sorted(code_list):
            if c not in code_set:
                continue
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.remove(c)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                changed = True
        if not changed:
            break
    return code_list


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Baseline: tetracode direct sum
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_idx = [_word_to_idx(w, q, n) for w in tc]
    else:
        best = []
        best_idx = []

    # Build neighbors
    neighbors, all_words = _build_neighbors(q, n, radius)

    try:
        # Greedy init (vectorized)
        greedy_deadline = min(deadline - 40.0, start_time + 60.0)
        code_list = _greedy_init(q, n, radius, neighbors, all_words, greedy_deadline)

        # Verify coverage
        if _verify_cover([list(all_words[i]) for i in code_list], q, n, radius, neighbors):
            # Local search
            ls_deadline = deadline - 10.0
            best_idx = _local_search(code_list, q, n, radius, neighbors, all_words, ls_deadline)
            # Polish
            polish_deadline = deadline - 3.0
            best_idx = _polish(best_idx, q, n, radius, neighbors, all_words, polish_deadline)
            # Convert to word lists
            candidate = [list(all_words[i]) for i in best_idx]
            # Final verification
            if _verify_cover(candidate, q, n, radius, neighbors):
                if len(candidate) < len(best):
                    best = candidate
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
