"""Forbidden-set tracking for A_2(18,6): maintain the set of all words within
Hamming distance < 6 of the current code. Each candidate check is O(1) set
membership. Uses multiple randomized restarts with different seeds and
deterministic weight-ordered passes."""
import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE_MARGIN = 5.0


def _reseeded_rng(seed):
    return random.Random(seed)


def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _all_words_within_radius(center, radius):
    """Generate all words within Hamming distance <= radius of center.
    For N=18 and radius=5, this is sum(C(18,k) for k=0..5) = 1+18+153+816+3060+8568 = 12616."""
    results = [center]
    bits = list(range(N))
    # Precompute combinations
    from itertools import combinations
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results


def _build_forbidden_set(words):
    """Build the forbidden set: all words within distance < MIN_DISTANCE of any codeword."""
    forbidden = set()
    for w in words:
        for neighbor in _all_words_within_radius(w, MIN_DISTANCE - 1):
            forbidden.add(neighbor)
    return forbidden


def _greedy_with_forbidden(deadline, seed=12345, order='random'):
    """Greedy construction using forbidden-set tracking."""
    words = _seed_words()
    forbidden = _build_forbidden_set(words)

    rng = _reseeded_rng(seed)

    if order == 'weight':
        # Deterministic weight-ordered pass
        all_candidates = list(range(2**N))
        all_candidates.sort(key=lambda x: x.bit_count())
        candidates = all_candidates
    else:
        # Randomized order
        candidates = list(range(2**N))
        rng.shuffle(candidates)

    for c in candidates:
        if time.time() > deadline:
            break
        if c not in forbidden:
            words.append(c)
            forbidden.update(_all_words_within_radius(c, MIN_DISTANCE - 1))

    return words


def _greedy_random_batch(deadline, seed=12345, batch_size=50000):
    """Greedy with random batch sampling and forbidden set."""
    words = _seed_words()
    forbidden = _build_forbidden_set(words)
    rng = _reseeded_rng(seed)
    total = 2**N

    while time.time() < deadline:
        # Sample a batch of random candidates
        samples = rng.sample(range(total), min(batch_size, total))
        for c in samples:
            if time.time() > deadline:
                break
            if c not in forbidden:
                words.append(c)
                forbidden.update(_all_words_within_radius(c, MIN_DISTANCE - 1))

    return words


def _greedy_weight_ordered(deadline):
    """Deterministic weight-ascending greedy with forbidden set."""
    words = _seed_words()
    forbidden = _build_forbidden_set(words)

    all_candidates = list(range(2**N))
    all_candidates.sort(key=lambda x: x.bit_count())

    for c in all_candidates:
        if time.time() > deadline:
            break
        if c not in forbidden:
            words.append(c)
            forbidden.update(_all_words_within_radius(c, MIN_DISTANCE - 1))

    return words


def _greedy_weight_desc(deadline):
    """Weight-descending greedy with forbidden set."""
    words = _seed_words()
    forbidden = _build_forbidden_set(words)

    all_candidates = list(range(2**N))
    all_candidates.sort(key=lambda x: x.bit_count(), reverse=True)

    for c in all_candidates:
        if time.time() > deadline:
            break
        if c not in forbidden:
            words.append(c)
            forbidden.update(_all_words_within_radius(c, MIN_DISTANCE - 1))

    return words


def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    # Strategy 1: Weight-ordered (deterministic, good baseline)
    if time.time() < deadline:
        try:
            w = _greedy_weight_ordered(deadline)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Strategy 2: Random restarts with different seeds
    for seed in [42, 123, 999, 7777, 31415, 27182, 16180, 8675309, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        try:
            w = _greedy_random_batch(deadline, seed=seed, batch_size=30000)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Strategy 3: Weight-descending
    if time.time() < deadline:
        try:
            w = _greedy_weight_desc(deadline)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Strategy 4: More random restarts if time remains
    for seed in range(100, 200):
        if time.time() > deadline:
            break
        try:
            w = _greedy_random_batch(deadline, seed=seed, batch_size=20000)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Fallback
    if best_words is None:
        best_words = _seed_words()

    # Ensure distinctness and range
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < 2**N:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
