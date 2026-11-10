# EVOLVE-BLOCK-START
import time
import itertools
import random

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _build_neighbors(q, n):
    """Return (all_words, word_idx, nrm) where nrm[i] is the list of indices
    of words within Hamming distance 1 of all_words[i]."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    nrm = []
    for w in all_words:
        nbrs = [word_idx[w]]
        for j in range(n):
            for d in (1, 2):
                b = list(w)
                b[j] = (b[j] + d) % q
                nbrs.append(word_idx[tuple(b)])
        nrm.append(nbrs)
    return all_words, word_idx, nrm


def _greedy_remove(base, q, n, radius):
    """Aggressive greedy removal: repeatedly remove any codeword that doesn't
    cause any word to become uncovered. This is the same as the safe-removal
    part of _improve_swap but done in a loop until no more safe removals exist."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors(q, n)

    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    sel_set = set(sel)
    deadline_guard = _HARD - 15.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        # Shuffle for diversity
        rnd_order = list(range(len(sel)))
        random.Random(42).shuffle(rnd_order)
        for idx in rnd_order:
            ci = sel[idx]
            removable = True
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    removable = False
                    break
            if removable:
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
        if not _deadline(15.0):
            break
    return [list(all_words[i]) for i in sel]


def _improve_swap(base, q, n, radius):
    """Time-guarded swap heuristic: try to replace codewords to reduce total count."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors(q, n)

    # Deduplicate and map to indices
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    # Coverage count per word
    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    sel_set = set(sel)
    deadline_guard = _HARD - 10.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        # Try to find a removable codeword whose removal can be repaired by a swap
        for ci in list(sel):
            # Find words that would become uncovered if we remove ci
            would_uncover = []
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    would_uncover.append(nb)
            if not would_uncover:
                # Safe removal: no word loses coverage
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
                break
            if len(would_uncover) > 3:
                continue
            # Try to find a replacement word that covers all would_uncover words
            # and is not already in sel
            # Candidate: any word that is a neighbor of all would_uncover words
            # For simplicity, try the would_uncover words themselves and their neighbors
            candidates = set()
            for uc in would_uncover:
                candidates.add(uc)
                for nb in nrm[uc]:
                    candidates.add(nb)
            best_cand = -1
            best_gain = 0
            for cand in candidates:
                if cand in sel_set:
                    continue
                # Check if cand covers all would_uncover words
                cand_nrm = nrm[cand]
                covers_all = True
                for uc in would_uncover:
                    if uc not in cand_nrm:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                # Gain: number of words where cov decreases after swap (i.e., words
                # that are covered by both ci and cand)
                # Actually, the gain in terms of enabling future removals is:
                # words that are in nrm[cand] but not in nrm[ci] get +1 cov,
                # words in nrm[ci] but not in nrm[cand] get -1 cov,
                # words in both get no change.
                # The net effect on "removability" is complex. Simple heuristic:
                # prefer candidates that cover many words with cov == 1 (to create
                # more redundancy) or many words with cov >= 2 (to reduce over-coverage).
                # Let's just do the swap if it covers all uncovered words.
                # The swap keeps the same size, so we need a different criterion.
                # Actually, the swap itself doesn't reduce size. We need to find
                # a swap that makes another codeword removable.
                # For now, just do the swap if it covers all would_uncover.
                best_cand = cand
                break
            if best_cand >= 0:
                # Perform swap: remove ci, add best_cand
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                for nb in nrm[best_cand]:
                    cov[nb] += 1
                sel.append(best_cand)
                sel_set.add(best_cand)
                improved = True
                break
        if not _deadline(10.0):
            break
    return [list(all_words[i]) for i in sel]


def _greedy_add(base, q, n, radius):
    """Time-guarded greedy set cover to reduce code size. Returns (base_list, covered_set)."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors(q, n)

    # map codeword list -> indices
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    # uncovered = words with cov == 0
    uncovered = [i for i in range(N) if cov[i] == 0]
    if not uncovered:
        return [list(all_words[i]) for i in sel]

    sel_set = set(sel)
    deadline_guard = _HARD - 20.0
    while uncovered and time.time() < deadline_guard:
        best_i = -1
        best_cov = 0
        # iterate candidates = uncovered set (they cover themselves + neighbors)
        for i in uncovered:
            c = 0
            for nb in nrm[i]:
                if cov[nb] == 0:
                    c += 1
            if c > best_cov:
                best_cov = c
                best_i = i
        if best_i < 0:
            break
        # add best_i
        sel.append(best_i)
        sel_set.add(best_i)
        for nb in nrm[best_i]:
            cov[nb] += 1
        # remove from uncovered
        uncovered = [i for i in uncovered if cov[i] == 0]
    return [list(all_words[i]) for i in sel]


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Build a guaranteed-valid base code ---
    base = _fallback_base(q, n, radius)

    # --- Guarantee validity: cover all words (repair if base was not perfect) ---
    if radius >= 1:
        N = q ** n
        all_words, word_idx, nrm = _build_neighbors(q, n)
        cov = [0] * N
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        uncovered = [i for i in range(N) if cov[i] == 0]
        # repair: add uncovered words as codewords
        while uncovered and _deadline(30.0):
            uc = uncovered.pop()
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                if uncovered:
                    uncovered = [i for i in uncovered if cov[i] == 0]

        base = [list(all_words[i]) for i in sel]

        # --- Greedy safe removal ---
        if _deadline(80.0) and len(base) <= 20000:
            base = _greedy_remove(base, q, n, radius)

        # --- Swap improvement ---
        if _deadline(50.0) and len(base) <= 20000:
            base = _improve_swap(base, q, n, radius)

        # --- Greedy add to ensure validity after swaps ---
        if _deadline(20.0) and len(base) <= 20000:
            base = _greedy_add(base, q, n, radius)

    return [list(c) for c in base]
# EVOLVE-BLOCK-END
