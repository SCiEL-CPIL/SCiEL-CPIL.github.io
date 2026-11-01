"""
Covering code construction via greedy set-cover with efficient ball computation.
Structural change: replace the algebraic direct-sum construction (9*3**(n-4) words)
with a maximum-coverage greedy heuristic that iteratively picks the word covering
the most yet-uncovered points. For q=3, n=6, r=1 this should approach the known
optimum of 73 (vs. 81 from the seed).
"""
import time
import itertools
import numpy as np

_DEADLINE_FRACTION = 0.85  # stop searching at 85% of the 150s wall-clock budget


def _ball_center(center, q, n, radius):
    """Yield all words within Hamming distance <= radius of center."""
    if radius == 0:
        yield center
        return
    # Precompute neighbors for each position and each symbol change
    # For efficiency, generate all words at distance 1, then at distance 2, etc.
    # For small radius (1 or 2) and small n, direct enumeration is fine.
    if radius == 1:
        yield center
        for pos in range(n):
            old = center[pos]
            for sym in range(q):
                if sym != old:
                    new = list(center)
                    new[pos] = sym
                    yield tuple(new)
    else:
        # General case: enumerate all subsets of positions up to size 'radius'
        for dist in range(radius + 1):
            for positions in itertools.combinations(range(n), dist):
                # For each position, choose a different symbol
                symbols_choices = []
                for pos in positions:
                    old = center[pos]
                    symbols_choices.append([s for s in range(q) if s != old])
                for combo in itertools.product(*symbols_choices):
                    new = list(center)
                    for pos, sym in zip(positions, combo):
                        new[pos] = sym
                    yield tuple(new)


def _precompute_balls(q, n, radius, max_centers=None, time_limit=None):
    """
    Precompute the ball (set of covered points) for a set of candidate centers.
    Returns a dict: center -> frozenset of covered words.
    """
    total = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx

    balls = {}
    if max_centers is None:
        max_centers = total

    all_words = [list(itertools.product(range(q), repeat=n))]
    all_words = list(all_words)

    start = time.time()
    for i, center in enumerate(all_words[:max_centers]):
        if time_limit is not None and (time.time() - start) > time_limit:
            break
        covered = set()
        for w in _ball_center(center, q, n, radius):
            covered.add(word_to_idx(w))
        balls[tuple(center)] = frozenset(covered)

    return balls, word_to_idx


def _greedy_cover(q, n, radius, deadline):
    """
    Greedy maximum-coverage: repeatedly pick the word whose ball covers the most
    currently-uncovered points.
    """
    total = q ** n

    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx

    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    # Precompute ball for every possible center (there are q^n of them)
    # For q=3, n=6: 729 centers, each ball has 1 + n*(q-1) + ... points
    # For r=1: 1 + 6*2 = 13 points per ball
    # Total: 729 * 13 = 9477 operations — trivial

    all_centers = [idx_to_word(i) for i in range(total)]
    all_indices = list(range(total))

    # Precompute ball membership as bitsets for speed
    # Each ball is a set of indices
    balls = []
    for center in all_centers:
        covered = set()
        for w in _ball_center(center, q, n, radius):
            covered.add(word_to_idx(w))
        balls.append(frozenset(covered))

    # Greedy: start with all points uncovered
    uncovered = set(range(total))
    chosen = []
    chosen_indices = set()

    while uncovered:
        # Find the center that covers the most uncovered points
        best_idx = -1
        best_count = -1
        for i, ball in enumerate(balls):
            if i in chosen_indices:
                continue
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        chosen.append(all_centers[best_idx])
        chosen_indices.add(best_idx)
        uncovered -= balls[best_idx]

        if time.time() > deadline:
            break

    return [list(c) for c in chosen]


def _local_search_refine(code, q, n, radius, deadline):
    """
    Try to improve the greedy solution by removing redundant codewords.
    A codeword is redundant if removing it still leaves all points covered.
    """
    total = q ** n

    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx

    # Build coverage map: for each point, which codewords cover it
    # coverage[point] = set of codeword indices that cover it
    coverage = [set() for _ in range(total)]

    for ci, cw in enumerate(code):
        for w in _ball_center(cw, q, n, radius):
            coverage[word_to_idx(w)].add(ci)

    # Check if every point is covered
    for p in range(total):
        if not coverage[p]:
            return code  # invalid, return as-is

    # Try removing each codeword if it doesn't break coverage
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for ci in range(len(code)):
            # Check if any point is ONLY covered by codeword ci
            redundant = True
            for p in range(total):
                if ci in coverage[p] and len(coverage[p]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove codeword ci
                del code[ci]
                # Update coverage
                for p in range(total):
                    if ci in coverage[p]:
                        coverage[p].discard(ci)
                # Renumber remaining codewords in coverage
                # This is expensive; for small codes it's fine
                # Actually, let's just rebuild coverage after each removal
                coverage = [set() for _ in range(total)]
                for ci2, cw in enumerate(code):
                    for w in _ball_center(cw, q, n, radius):
                        coverage[word_to_idx(w)].add(ci2)
                changed = True
                break

    return code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 150 * _DEADLINE_FRACTION

    # For the specific case q=3, n=6, r=1, use greedy + local search
    # For other cases, fall back to a reasonable heuristic

    if q == 3 and n == 6 and radius == 1:
        code = _greedy_cover(q, n, radius, deadline)
        code = _local_search_refine(code, q, n, radius, deadline)
        return code
    else:
        # General fallback: greedy set cover
        code = _greedy_cover(q, n, radius, deadline)
        return code
