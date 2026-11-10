"""Covering code K_q(n, radius): greedy construction with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 140.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    """Return a uint8 array of shape (q**n, n) containing every word in {0..q-1}^n."""
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_greedy(q, n, radius, seed):
    """Greedy maximal covering: repeatedly add the word covering the most uncovered words.

    Returns (codewords, covered_mask, words).
    """
    rng = random.Random(seed)
    words = _all_words(q, n)
    total = words.shape[0]

    # Precompute the list of positions within distance <= radius of each word.
    # A word at index idx has neighbours = all words differing in <= radius coordinates.
    # Build via base-q digits.
    # For each word, generate all words at Hamming distance 0..radius.
    # We precompute per-coordinate "sets" but that's huge; instead compute on the fly
    # using the fact that a neighbour differs in a subset S of coordinates (|S|<=radius),
    # and for each coordinate in S the value is in 1..q-1 alternatives.

    # Precompute, for each coordinate i, the list of index-offsets for each symbol change.
    # Changing coordinate i from value v to v' changes the flat index by (v'-v)*q**(n-1-i).
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # For a given word index idx, the set of neighbours = union over subsets S, |S|<=radius
    # of product of (idx + (v'-v)*stride[i]) for i in S, v' in 1..q-1.
    # This is expensive per candidate. We limit candidates to a random sample each round.

    covered = np.zeros(total, dtype=bool)
    codewords = []

    # Precompute all "delta patterns": for each subset of coordinates and each
    # non-zero symbol change per coordinate, the flat offset.
    # Group by subset size.
    # For efficiency, precompute for each radius-level r the list of offset-tuples.
    # An offset-tuple is a list of (coord, new_symbol) -> but we need the resulting index.
    # Better: precompute for each word the list of neighbour indices? That's total * (sum C(n,k)(q-1)^k)
    # which for n=7,q=3,r=1 is 2187 * (1 + 7*2) = 2187*15 = 32805 entries. Very small!

    # So precompute neighbour lists for ALL words.
    # neighbour_indices[word] = list of flat indices within distance <= radius.
    # Build using digits.
    words_int = words.astype(np.int64)
    # For each word, digits are words_int[word].
    # For radius 1: neighbours = self + for each coord i, for each v' != v, word with coord i = v'.

    # General: enumerate all subsets of coordinates of size <= radius, and for each
    # coordinate in the subset choose a new symbol in 1..q-1.
    # Precompute the set of "offset vectors" as arrays of shape (num_patterns, n)
    # where each row is the flat-index offset (can be negative).
    # Then neighbour index = idx + offset.
    # But offset depends on current symbol? No: changing coord i from v to v' changes
    # index by (v'-v)*stride[i]. The offset (v'-v) depends on v. So we can't precompute
    # a single offset list independent of the word.

    # Alternative: precompute for each word the neighbour list by digit manipulation.
    # For n=7, q=3, r=1: 15 neighbours per word. 2187 words. Total 32805. Fine.
    # For general q,n,radius, this could be large, but the task is fixed at q=3,n=7,r=1
    # (or small). Let's handle the general case but cap.

    num_patterns = 1  # the word itself
    # For radius 1: + n*(q-1)
    # For radius 2: + C(n,2)*(q-1)^2
    # etc.
    # Precompute neighbour lists for all words.
    # Represent each word's digits. For each subset of coords size k<=radius, and each
    # assignment of new symbols, compute the new index.

    # For efficiency, precompute the list of (coord, new_symbol) combinations for each
    # subset size, then for a given word compute the new index.

    # Build neighbour lists.
    # For radius 1:
    if radius == 1:
        # neighbours[word] = [word] + for i in range(n): for v' in 0..q-1, v'!=v: new_word
        # Precompute as a list of lists.
        nb = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
        # Actually variable length per word? No, always 1 + n*(q-1) for radius 1.
        # Fill.
        for idx in range(total):
            digits = words_int[idx]
            pos = 0
            nb[idx, pos] = idx
            pos += 1
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    new_idx = idx + (vp - v) * stride[i]
                    nb[idx, pos] = new_idx
                    pos += 1
    else:
        # General radius: enumerate all patterns.
        # This is more complex; for the task radius=1 is expected. Fallback to radius-1 logic
        # if radius>1 by iterating.
        # For now, handle radius=1 only efficiently; else use a slower method.
        # Build patterns: list of (list of (coord, new_symbol)) for all changes with total
        # coords changed <= radius.
        # For a word, new_idx = idx + sum over (coord, new_sym) of (new_sym - digit[coord])*stride[coord].
        # But new_sym - digit[coord] depends on digit. So we store (coord, new_sym) and compute.
        # Enumerate all subsets of coords of size k=0..radius, and for each, all (q-1)^k
        # new-symbol assignments.
        patterns = []  # list of lists of (coord, new_sym)
        for k in range(radius + 1):
            for subset in _combinations(range(n), k):
                # all assignments of new symbols (1..q-1 alternatives each)
                # For each coord in subset, new_sym in 0..q-1 excluding the current digit
                # (which we don't know yet). So store (coord, new_sym) for new_sym in 0..q-1
                # and later skip if new_sym == digit[coord].
                # This over-generates but we filter per word.
                # For k=0: empty pattern (the word itself).
                if k == 0:
                    patterns.append([])
                else:
                    for assign in _product(range(q), repeat=k):
                        # assign[j] is the new symbol for subset[j]
                        patterns.append([(subset[j], assign[j]) for j in range(k)])
        # Now for each word, compute neighbours by applying each pattern and filtering
        # out patterns where new_sym == digit[coord] (which would mean no change).
        # Actually if new_sym == digit[coord], the coordinate is unchanged, so the pattern
        # effectively changes fewer coordinates. That's fine (still within radius).
        # But we might generate duplicates. For correctness, duplicates are OK (we just
        # mark covered).
        # Number of patterns = sum_{k=0}^{radius} C(n,k) * q^k. For n=7,q=3,r=1: 1 + 7*3 = 22.
        # For each word, 22 candidates. 2187*22 = 48114. Fine.
        nb_list = []
        for idx in range(total):
            digits = words_int[idx]
            nbs = []
            for pat in patterns:
                new_idx = idx
                valid = True
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                # new_idx should be in [0, total)
                if 0 <= new_idx < total:
                    nbs.append(new_idx)
            nb_list.append(nbs)
        # Use nb_list (list of lists) instead of a 2D array.
        nb = nb_list

    # Greedy: repeatedly pick the candidate (from uncovered words) whose neighbour set
    # covers the most uncovered words. Add it, mark covered.
    # For efficiency, maintain for each word a count of uncovered neighbours.
    # When we add a codeword, we mark its neighbours covered, and for each other word
    # whose neighbour set includes any of those, decrement its count.
    # This is the standard greedy set-cover.
    # uncovered_count[word] = number of uncovered words in nb[word].
    # Initially uncovered_count[word] = len(nb[word]) (all uncovered).
    # When we cover a set of words C, for each word w, uncovered_count[w] -= |C ∩ nb[w]|.
    # We need to efficiently update. For each newly covered word c, for each word w that
    # has c in nb[w], decrement uncovered_count[w]. So we need the reverse mapping:
    # for each word c, the list of words w such that c in nb[w]. By symmetry of Hamming
    # distance, c in nb[w] iff w in nb[c]. So reverse_nb[c] = nb[c] (same set).
    # So when we cover c, for each w in nb[c], decrement uncovered_count[w].

    uncovered_count = np.array([len(x) if isinstance(nb, list) else len(nb[i]) for i, x in
                                enumerate(nb if isinstance(nb, list) else [None]*total)],
                               dtype=np.int64)
    # Actually simpler: uncovered_count[i] = number of uncovered in nb[i].
    # Initially all uncovered, so = len(nb[i]).
    if isinstance(nb, list):
        uncovered_count = np.array([len(nb[i]) for i in range(total)], dtype=np.int64)
    else:
        uncovered_count = np.full(total, nb.shape[1], dtype=np.int64)

    while True:
        if time.time() > _DEADLINE - 10:
            break
        # Pick the word with max uncovered_count among uncovered words.
        # Restrict to uncovered words (covered words are not candidates since they're
        # already covered, but adding a covered word could still help? No, a covered word
        # is already within radius of some codeword, so adding it doesn't cover new words
        # that aren't already... actually it might cover words not covered by the existing
        # codeword. But greedy set cover typically picks from all sets. Let's pick from
        # all words but prefer uncovered. Actually the standard approach: pick the set
        # (word) that covers the most uncovered elements. A word that is itself covered
        # can still cover other uncovered words. So we should consider all words.
        # But to reduce search, consider all words.
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] == 0:
            break
        # Add best_idx as a codeword.
        codewords.append(words[best_idx].tolist())
        # Mark its neighbours as covered, and update uncovered_count.
        if isinstance(nb, list):
            nbs = nb[best_idx]
        else:
            nbs = nb[best_idx]
        for c in nbs:
            if not covered[c]:
                covered[c] = True
                # For each w in nb[c], decrement uncovered_count[w].
                if isinstance(nb, list):
                    for w in nb[c]:
                        uncovered_count[w] -= 1
                else:
                    for w in nb[c]:
                        uncovered_count[w] -= 1
        # Also decrement for the codeword itself (it's in its own nb).
        # Already handled since best_idx is in nbs.

    return codewords, covered, words


def _combinations(iterable, r):
    """Simple combinations generator (avoid itertools for clarity, but use itertools)."""
    import itertools
    return itertools.combinations(iterable, r)


def _product(iterable, repeat):
    import itertools
    return itertools.product(iterable, repeat=repeat)


def construct(q, n, radius):
    start = time.time()

    # For the specific case q=3, n=7, radius=1, use the greedy construction.
    # Fallback to the direct-sum construction for other parameters or on timeout.
    try:
        codewords, covered, words = _build_greedy(q, n, radius, seed=42)
        # Verify coverage.
        if covered.all():
            return codewords
        # If not fully covered (shouldn't happen with greedy), fall through.
    except Exception:
        pass

    # Fallback: direct sum of ternary tetracode.
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # For n < 4, just return all words (trivial covering).
        return [list(w) for w in itertools.product(range(3), repeat=n)]
# EVOLVE-BLOCK-END
