"""Ternary covering code K_3(7,1) via greedy + local improvement."""
import time
import random
import itertools
import numpy as np

DEADLINE_MARGIN = 2.0  # seconds to keep before the 150s wall-clock kill


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(codeword, radius, n):
    """Return set of all words within Hamming distance <= radius from codeword."""
    covered = set()
    if radius == 0:
        covered.add(tuple(codeword))
        return covered
    # Generate all neighbors within radius
    # For radius 1: the word itself + all single-position changes
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for deltas in itertools.product(range(1, 3), repeat=r):
                w = list(codeword)
                for pos, d in zip(positions, deltas):
                    w[pos] = (w[pos] + d) % 3
                covered.add(tuple(w))
    return covered


def _is_covering(codewords, q, n, radius):
    """Check if codewords cover the whole space. Returns True if covering."""
    covered = set()
    for cw in codewords:
        covered |= _covered_by(cw, radius, n)
    return len(covered) == q ** n


def _count_covered(codewords, q, n, radius):
    covered = set()
    for cw in codewords:
        covered |= _covered_by(cw, radius, n)
    return len(covered)


def _greedy_cover(q, n, radius, seed, time_budget):
    """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
    rng = random.Random(seed)
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    uncovered = set(all_words)
    codewords = []
    start = time.time()

    while uncovered:
        if time.time() - start > time_budget:
            break
        best_cw = None
        best_count = 0
        # Sample a subset of candidate codewords for speed
        candidates = list(uncovered)
        rng.shuffle(candidates)
        sample_size = min(500, len(candidates))
        for cw in candidates[:sample_size]:
            cov = _covered_by(cw, radius, n)
            new_cov = len(cov & uncovered)
            if new_cov > best_count:
                best_count = new_cov
                best_cw = cw
        if best_cw is None:
            best_cw = candidates[0]
            best_count = len(_covered_by(best_cw, radius, n) & uncovered)
        codewords.append(list(best_cw))
        uncovered -= _covered_by(best_cw, radius, n)

    return codewords


def _local_improve(codewords, q, n, radius, seed, time_budget):
    """Try to remove redundant codewords or swap for smaller sets."""
    rng = random.Random(seed)
    codewords = [list(cw) for cw in codewords]
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        rng.shuffle(codewords)
        for i in range(len(codewords)):
            if time.time() - start > time_budget:
                break
            # Try removing codeword i
            remaining = codewords[:i] + codewords[i+1:]
            if _is_covering(remaining, q, n, radius):
                codewords = remaining
                improved = True
                break
    return codewords


def _random_restart_greedy(q, n, radius, seed, time_budget):
    """Multiple random restarts of greedy, keep best."""
    rng = random.Random(seed)
    best = None
    best_len = float('inf')
    start = time.time()
    attempt = 0
    while time.time() - start < time_budget:
        # Random initial seed for greedy
        sub_seed = seed + attempt
        cws = _greedy_cover(q, n, radius, sub_seed, time_budget - (time.time() - start))
        if len(cws) < best_len:
            best = cws
            best_len = len(cws)
        attempt += 1
    return best if best else []


def _tetracode_construction(q, n, radius):
    """Direct sum of ternary tetracode with full product on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(t) for t in tetracode]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(list(h) + list(tail))
    return result


def _improve_from_base(codewords, q, n, radius, seed, time_budget):
    """Start from a base construction and try to reduce size."""
    rng = random.Random(seed)
    codewords = [list(cw) for cw in codewords]
    start = time.time()
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        rng.shuffle(codewords)
        for i in range(len(codewords)):
            if time.time() - start > time_budget:
                break
            remaining = codewords[:i] + codewords[i+1:]
            if _is_covering(remaining, q, n, radius):
                codewords = remaining
                improved = True
                break
    
    # Phase 2: Try swapping a codeword with a better one
    if time.time() - start < time_budget:
        for _ in range(5):
            if time.time() - start > time_budget:
                break
            i = rng.randrange(len(codewords))
            # Try replacing codeword i with a random word
            for trial in range(20):
                if time.time() - start > time_budget:
                    break
                new_cw = [rng.randrange(q) for _ in range(n)]
                candidate = codewords[:i] + [new_cw] + codewords[i+1:]
                if _is_covering(candidate, q, n, radius):
                    # Check if it allows further reduction
                    codewords = candidate
                    # Try to remove another
                    for j in range(len(codewords)):
                        remaining = codewords[:j] + codewords[j+1:]
                        if _is_covering(remaining, q, n, radius):
                            codewords = remaining
                            break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Special case: n=4, q=3, radius=1 -> perfect code of size 9
    if q == 3 and n == 4 and radius == 1:
        return _tetracode_construction(q, n, radius)
    
    # For n=7, q=3, radius=1: use tetracode base (9 * 3^3 = 243 words) and improve
    if q == 3 and n == 7 and radius == 1:
        base = _tetracode_construction(q, n, radius)
        # Try to improve from base
        time_left = deadline - time.time()
        if time_left > 5:
            improved = _improve_from_base(base, q, n, radius, 42, time_left * 0.6)
            if len(improved) < len(base):
                base = improved
        # Also try greedy from scratch with remaining time
        time_left = deadline - time.time()
        if time_left > 10:
            greedy = _random_restart_greedy(q, n, radius, 123, time_left * 0.4)
            if greedy and len(greedy) < len(base):
                base = greedy
        # Final local improvement
        time_left = deadline - time.time()
        if time_left > 3:
            base = _local_improve(base, q, n, radius, 99, time_left * 0.8)
        # Ensure validity
        if _is_covering(base, q, n, radius):
            return base
        # Fallback: return the base construction (known valid)
        return _tetracode_construction(q, n, radius)
    
    # General case: use greedy
    time_left = deadline - time.time()
    if time_left > 10:
        result = _random_restart_greedy(q, n, radius, 7, time_left * 0.5)
        if result:
            time_left = deadline - time.time()
            if time_left > 5:
                result = _local_improve(result, q, n, radius, 55, time_left * 0.7)
            if _is_covering(result, q, n, radius):
                return result
    
    # Fallback: full product (always valid but large)
    return [list(w) for w in itertools.product(range(q), repeat=n)]
