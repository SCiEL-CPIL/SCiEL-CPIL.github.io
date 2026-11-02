"""Ternary covering code K_3(7,1) via greedy set cover with local search."""
import time
import itertools
import random

# Deadline guard: 150s wall clock, keep 15s safety margin
_DEADLINE = time.time() + 130.0


def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of same length."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _neighbors(word, q, n):
    """Generate all words within Hamming distance 1 of word."""
    yield word
    for pos in range(n):
        old = word[pos]
        for val in range(q):
            if val != old:
                new = list(word)
                new[pos] = val
                yield tuple(new)


def _build_incidence(q, n, radius, codewords):
    """
    For each codeword, compute the set of words it covers (within Hamming distance <= radius).
    Returns a list of sets (frozensets) of covered word indices.
    Also returns a dict mapping each word index to the list of codeword indices covering it.
    """
    # Total number of words
    total = q ** n
    # Map word tuple to index
    word_to_idx = {}
    idx_to_word = {}
    for idx in range(total):
        w = tuple(int(d) for d in format(idx, '0{}b'.format(n * 2)))  # placeholder, will fix
    # Better: enumerate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = all_words

    # For radius 1, each codeword covers 1 + n*(q-1) words
    coverage = []
    for cw in codewords:
        cw_tuple = tuple(cw)
        covered = set()
        covered.add(word_to_idx[cw_tuple])
        for pos in range(n):
            old = cw_tuple[pos]
            for val in range(q):
                if val != old:
                    new = list(cw_tuple)
                    new[pos] = val
                    covered.add(word_to_idx[tuple(new)])
        coverage.append(covered)

    return coverage, all_words, word_to_idx


def _greedy_cover(q, n, radius, time_limit=60.0):
    """
    Greedy set cover: repeatedly pick the codeword that covers the most uncovered words.
    """
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute for each possible codeword, the set of words it covers
    # This is expensive for large spaces, but for q=3, n=7, radius=1: each covers 1+7*2=15 words
    # Total codeword candidates: 3^7 = 2187
    # We can precompute coverage for all 2187 possible codewords
    start = time.time()
    candidate_cw = all_words  # all possible codewords
    coverage_map = {}
    for cw in candidate_cw:
        covered = set()
        covered.add(word_to_idx[cw])
        for pos in range(n):
            old = cw[pos]
            for val in range(q):
                if val != old:
                    new = list(cw)
                    new[pos] = val
                    covered.add(word_to_idx[tuple(new)])
        coverage_map[cw] = covered

    # Greedy selection
    uncovered = set(range(total))
    selected = []
    selected_set = set()

    while uncovered and time.time() < start + time_limit:
        best_cw = None
        best_gain = 0
        for cw in candidate_cw:
            if cw in selected_set:
                continue
            gain = len(coverage_map[cw] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cw = cw
        if best_cw is None or best_gain == 0:
            break
        selected.append(best_cw)
        selected_set.add(best_cw)
        uncovered -= coverage_map[best_cw]

    return [list(cw) for cw in selected]


def _local_search(codewords, q, n, radius, time_limit=30.0):
    """
    Try to remove redundant codewords, then try to swap codewords to reduce count.
    """
    start = time.time()
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    def covers(word, cw):
        """Check if word is within radius of codeword cw."""
        dist = sum(1 for a, b in zip(word, cw) if a != b)
        return dist <= radius

    def is_covered(word, cw_list):
        for cw in cw_list:
            if covers(word, cw):
                return True
        return False

    # Try removing each codeword and checking if all words are still covered
    current = [list(cw) for cw in codewords]
    improved = True
    while improved and time.time() < start + time_limit:
        improved = False
        for i in range(len(current)):
            if time.time() >= start + time_limit:
                break
            candidate = current[:i] + current[i+1:]
            if not candidate:
                continue
            # Check if all words are still covered
            all_covered = True
            for w in all_words:
                if not is_covered(w, candidate):
                    all_covered = False
                    break
            if all_covered:
                current = candidate
                improved = True
                break  # restart from beginning since list changed

    return current


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius radius.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    global _DEADLINE
    start_time = time.time()
    deadline = start_time + 130.0  # 150s limit, keep 20s margin

    # For the specific case q=3, n=7, radius=1, use a targeted approach
    if q == 3 and n == 7 and radius == 1:
        # Greedy set cover
        greedy_start = time.time()
        greedy_limit = min(60.0, deadline - time.time() - 30.0)
        if greedy_limit < 5.0:
            # Not enough time, use direct sum construction as fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

        codewords = _greedy_cover(q, n, radius, time_limit=greedy_limit)

        # Local search to try to reduce further
        remaining = deadline - time.time() - 10.0
        if remaining > 5.0 and codewords:
            codewords = _local_search(codewords, q, n, radius, time_limit=remaining)
        return codewords

    # General fallback: direct sum or greedy
    if time.time() > deadline:
        # Fallback: trivial construction
        return [[0] * n]

    # For small cases, use greedy
    if q ** n <= 10000:
        return _greedy_cover(q, n, radius, time_limit=min(60.0, deadline - time.time() - 10.0))

    # Fallback: direct sum for ternary
    if q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Last resort: all zeros
    return [[0] * n]
