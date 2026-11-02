"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 120.0
_MARGIN = 15.0


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _init_solution(q, n):
    """Direct sum construction: 9 * 3^(n-4) codewords."""
    tet = _tetracode()
    code = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(h + list(tail))
    return code


def _covered_set(code, n, q):
    """Return a boolean array of size q^n indicating which words are covered (dist <= 1)."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in code:
        # center
        idx = 0
        for i, v in enumerate(cw):
            idx = idx * q + v
        covered[idx] = True
        # radius-1 neighbors
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                idx = 0
                for i in range(n):
                    v = val if i == pos else cw[i]
                    idx = idx * q + v
                covered[idx] = True
    return covered


def _uncovered_sample(covered, q, n, max_samples=200000, rng=None):
    """Sample some uncovered words from the full space."""
    if rng is None:
        rng = random
    total = q ** n
    # Find uncovered indices
    uncovered_idx = np.where(~covered)[0]
    if len(uncovered_idx) == 0:
        return []
    if len(uncovered_idx) <= max_samples:
        sample_idx = uncovered_idx
    else:
        sample_idx = uncovered_idx[rng.sample(range(len(uncovered_idx)), max_samples)]
    # Convert to base-q vectors
    words = []
    for idx in sample_idx:
        w = []
        for i in range(n):
            w.append(int(idx % q))
            idx //= q
        w.reverse()
        words.append(w)
    return words


def _greedy_add(code, n, q, rng, time_limit):
    """
    Greedy: repeatedly pick the codeword that covers the most uncovered words.
    Use sampling for efficiency.
    """
    start = time.time()
    covered = _covered_set(code, n, q)
    uncovered = ~covered
    total_uncovered = int(uncovered.sum())

    if total_uncovered == 0:
        return code

    # Precompute: for each possible position and value, how many uncovered words are at that offset
    # We'll use a sampling-based greedy
    while total_uncovered > 0 and time.time() - start < time_limit:
        # Sample uncovered words to estimate gains
        samples = _uncovered_sample(covered, q, n, max_samples=50000, rng=rng)
        if not samples:
            break

        # For each candidate codeword (from a pool), count how many samples it covers
        # Candidate pool: all words at distance 1 from sampled words
        candidate_counts = {}
        for s in samples:
            # The sample itself
            key = tuple(s)
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
            # Distance-1 neighbors of the sample
            for pos in range(n):
                orig = s[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    ns = list(s)
                    ns[pos] = val
                    key = tuple(ns)
                    candidate_counts[key] = candidate_counts.get(key, 0) + 1

        # Pick the best candidate
        if not candidate_counts:
            break
        best_cand = max(candidate_counts, key=candidate_counts.get)
        best_cand = list(best_cand)

        # Verify it actually covers something new
        # Add it
        code.append(best_cand)
        # Update covered
        idx = 0
        for i, v in enumerate(best_cand):
            idx = idx * q + v
        covered[idx] = True
        for pos in range(n):
            orig = best_cand[pos]
            for val in range(q):
                if val == orig:
                    continue
                idx = 0
                for i in range(n):
                    v = val if i == pos else best_cand[i]
                    idx = idx * q + v
                if not covered[idx]:
                    covered[idx] = True
                    total_uncovered -= 1

    return code


def _remove_redundant(code, n, q, time_limit):
    """
    Try to remove codewords that don't contribute uniquely to coverage.
    For each codeword, check if all words it uniquely covers are also covered by others.
    """
    start = time.time()
    code = list(code)
    m = len(code)
    if m == 0:
        return code

    # Build coverage: for each word in space, which codewords cover it
    # This is expensive for large spaces, so we use a smarter approach
    # For each codeword c, find words that ONLY c covers (among the current set)
    # A word w is uniquely covered by c if dist(w,c)<=1 and for all other c', dist(w,c')>1

    # Efficient approach: for each codeword c, its sphere is {c} U {words at dist 1}
    # A word w in sphere(c) is NOT uniquely covered by c if there exists c' != c with dist(w,c')<=1
    # dist(w,c')<=1 means w is in sphere(c')
    # So w is uniquely covered by c iff w is in sphere(c) and w is NOT in sphere(c') for any c'!=c

    # We can check: for each c, look at its sphere. For each w in sphere(c),
    # check if w is in the sphere of any other codeword.
    # Two codewords c1, c2 have overlapping spheres iff dist(c1,c2) <= 2.

    # For each codeword c, find all other codewords c' with dist(c,c')<=2.
    # Then c's unique words are sphere(c) minus union of sphere(c') for those c'.

    # This is still expensive. Let's use a simpler heuristic:
    # For each codeword c, pick a few random words in its sphere and check if they're covered by others.
    # If all checked words are covered by others, try removing c.

    rng = random.Random(42)
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        new_code = []
        for i, c in enumerate(code):
            # Check if c can be removed
            # Sample words in sphere(c)
            sphere_words = [list(c)]
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w = list(c)
                    w[pos] = val
                    sphere_words.append(w)

            # Check if all these words are covered by code without c
            can_remove = True
            for w in sphere_words:
                covered_by_other = False
                for j, c2 in enumerate(code):
                    if j == i:
                        continue
                    # dist(w, c2) <= 1?
                    d = 0
                    for k in range(n):
                        if w[k] != c2[k]:
                            d += 1
                            if d > 1:
                                break
                    if d <= 1:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    can_remove = False
                    break

            if can_remove:
                changed = True
                # Don't add c to new_code
            else:
                new_code.append(c)
        code = new_code

    return code


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius `radius`.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    # Initial solution from direct sum
    code = _init_solution(q, n)

    # Greedy improvement
    rng = random.Random(12345)
    remaining = deadline - time.time()
    if remaining > 5:
        code = _greedy_add(code, n, q, rng, remaining - 5)

    # Remove redundant codewords
    remaining = deadline - time.time()
    if remaining > 3:
        code = _remove_redundant(code, n, q, remaining - 3)

    # Final safety check: ensure we're within deadline
    if time.time() > deadline:
        pass  # Just return what we have

    return code
# EVOLVE-BLOCK-END
