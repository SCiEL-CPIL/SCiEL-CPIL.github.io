"""Combined program: fast incremental-gain greedy + deterministic peeling + neighbor-restricted
swaps + symmetry-aware restart, specialized for the ternary K_3(8,1) covering code."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _build_neighbors(q, n, radius):
    """Return (neighbors, all_words). neighbors[i] = list of indices within Hamming distance
    <= radius of word i (includes i). all_words[i] = the word as a list."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)
    neighbors = [None] * total
    for idx in range(total):
        w = all_words[idx]
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w[pos] = val
                    nbrs.append(int(np.dot(w, q ** np.arange(n - 1, -1, -1, dtype=np.int64))))
                w[pos] = orig
        neighbors[idx] = nbrs
    return neighbors, all_words


def _tetracode_solution(q, n, radius):
    """Direct-sum tetracode core: 9 * q^(n-4) words (valid for q=3, n>=4, radius>=1)."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_init(neighbors, all_words, deadline):
    """Fast incremental-gain greedy set cover. Returns list of codeword indices covering all."""
    total = len(neighbors)
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break
        # newly covered count for every candidate = number of currently uncovered neighbors
        gains = np.zeros(total, dtype=np.int32)
        for nbrs in neighbors:
            gains[nbrs] += 1
        gains[uncovered == 0] = 0
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        code.append(best)
        uncovered[neighbors[best]] = 0
    return code


def _verify(code_idx, neighbors, all_words):
    total = len(neighbors)
    covered = np.zeros(total, dtype=np.int8)
    for c in code_idx:
        covered[neighbors[c]] = 1
    return bool(covered.all())


def _peel(code_idx, neighbors, deadline):
    """Deterministic peeling: repeatedly remove any codeword whose removal keeps coverage.
    Returns (code_list, cover_count)."""
    total = len(neighbors)
    code_list = list(code_idx)
    code_set = set(code_list)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors[c]] += 1
    for _ in range(20):
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
                cover_count[neighbors[c]] -= 1
                changed = True
        if not changed:
            break
    return code_list, cover_count


def _local_search(code_idx, neighbors, all_words, deadline, seed=12345):
    """Peel + neighbor-restricted swaps. Returns best code list (indices)."""
    rng = random.Random(seed)
    total = len(neighbors)
    code_list, cover_count = _peel(code_idx, neighbors, deadline)
    code_set = set(code_list)
    best_size = len(code_list)
    best_code = list(code_list)

    while time.time() < deadline:
        # ---- peel pass ----
        code_list, cover_count = _peel(code_list, neighbors, deadline)
        code_set = set(code_list)
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)

        if time.time() > deadline:
            break
        if best_size <= 450:
            break

        # ---- swap move: remove one, replace with best neighbor of its uncovered words ----
        c = rng.choice(code_list)
        cover_count[neighbors[c]] -= 1
        code_list.remove(c)
        code_set.discard(c)
        uncovered = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        if not uncovered:
            continue  # pure removal, keep it out

        # candidates restricted to neighbors of the uncovered words (symmetry-aware, fast)
        cand = set()
        for u in uncovered:
            cand.update(neighbors[u])
        cand.discard(c)
        best_cand = -1
        best_gain = 0
        for cand_i in cand:
            gain = 0
            for nbr in neighbors[cand_i]:
                if cover_count[nbr] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = cand_i
        if best_cand >= 0 and best_gain > 0:
            code_list.append(best_cand)
            code_set.add(best_cand)
            cover_count[neighbors[best_cand]] += 1
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        else:
            # restore
            code_list.append(c)
            code_set.add(c)
            cover_count[neighbors[c]] += 1

    return best_code


def _ternary_8_1():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    q, n, radius = 3, 8, 1
    neighbors, all_words = _build_neighbors(q, n, radius)

    # ---- build a good starting code: tetracode direct sum, then greedy-fill any gaps ----
    tc = _tetracode_solution(q, n, radius)
    total = len(neighbors)
    covered = np.zeros(total, dtype=np.int8)
    base = []
    for w in tc:
        idx = int(np.dot(w, q ** np.arange(n - 1, -1, -1, dtype=np.int64)))
        base.append(idx)
        covered[neighbors[idx]] = 1
    remaining = np.zeros(total, dtype=np.int8)
    remaining[covered == 0] = 1
    # fill gaps with incremental greedy
    code = list(base)
    while remaining.any():
        if time.time() > deadline - 30.0:
            break
        gains = np.zeros(total, dtype=np.int32)
        for nbrs in neighbors:
            gains[nbrs] += 1
        gains[remaining == 0] = 0
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        code.append(best)
        remaining[neighbors[best]] = 0
    # if greedy couldn't finish (deadline), fall back to full greedy
    if remaining.any():
        code = _greedy_init(neighbors, all_words, deadline - 20.0)

    # ---- local search with a couple of restarts to break symmetry ----
    best = code
    for seed in (12345, 777):
        ls_deadline = min(deadline - 8.0, start + 140.0)
        cand = _local_search(code, neighbors, all_words, ls_deadline, seed=seed)
        if _verify(cand, neighbors, all_words) and len(cand) < len(best):
            best = cand
        if time.time() > deadline - 6.0:
            break

    # ---- final deterministic peel polish ----
    polished, _ = _peel(best, neighbors, deadline - 2.0)
    if _verify(polished, neighbors, all_words) and len(polished) < len(best):
        best = polished

    return [list(all_words[i]) for i in best]


def _fallback_construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0
    neighbors, all_words = _build_neighbors(q, n, radius)
    code = _greedy_init(neighbors, all_words, deadline - 15.0)
    if not _verify(code, neighbors, all_words):
        return [list(all_words[i]) for i in range(len(neighbors))]  # trivial full cover
    cand = _local_search(code, neighbors, all_words, deadline - 8.0)
    if _verify(cand, neighbors, all_words) and len(cand) < len(code):
        code = cand
    return [list(all_words[i]) for i in code]


def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        try:
            return _ternary_8_1()
        except Exception:
            pass
    return _fallback_construct(q, n, radius)
# EVOLVE-BLOCK-END
