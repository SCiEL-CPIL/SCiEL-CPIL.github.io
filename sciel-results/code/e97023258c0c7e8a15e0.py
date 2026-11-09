"""Superpermutation construction with time-bounded local shortening."""
import time
from itertools import permutations

# Hard safety margin below the evaluator's 150s wall-clock limit.
DEADLINE = time.time() + 140.0


def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _counts(word, n):
    """Return dict mapping each n-permutation -> occurrence count in word."""
    from collections import defaultdict
    c = defaultdict(int)
    for i in range(len(word) - n + 1):
        c[tuple(word[i:i + n])] += 1
    return c


def _all_perms(n):
    return [tuple(p) for p in permutations(range(1, n + 1))]


def construct(n):
    if n <= 1:
        return "1"

    start = time.time()
    word = _build_recursive(n)
    allp = set(_all_perms(n))

    if n <= 5:
        # Small enough that the recursive form is already optimal; no search needed.
        return "".join(map(str, word))

    # Rolling-hash counters for fast "is every permutation still present" checks.
    base = n + 1  # symbols are 1..n, so base > max symbol

    def roll_counts(w):
        h = 0
        for x in w[:n]:
            h = h * base + x
        seen = {h: 1}
        for i in range(n, len(w)):
            h = (h - w[i - n] * (base ** (n - 1))) * base + w[i]
            seen[h] = seen.get(h, 0) + 1
        return seen

    # Precompute base**(n-1) for the roll.
    powb = base ** (n - 1)

    def roll_counts2(w):
        h = 0
        for x in w[:n]:
            h = h * base + x
        seen = {h: 1}
        for i in range(n, len(w)):
            h = (h - w[i - n] * powb) * base + w[i]
            seen[h] = seen.get(h, 0) + 1
        return seen

    # Map each hash value to the permutation it represents (for validity vs allp).
    hash_to_perm = {}
    for p in allp:
        h = 0
        for x in p:
            h = h * base + x
        hash_to_perm[h] = p

    # A word is valid iff the set of hashes present == set of all permutation hashes.
    target_hashes = set(hash_to_perm.keys())

    def is_valid_counts(seen):
        # every target hash must appear at least once
        for h in target_hashes:
            if h not in seen:
                return False
        return True

    # Baseline validity check (should be True).
    base_seen = roll_counts2(word)
    # (Sanity: trust recursive construction; skip expensive per-hash set build.)

    # Local search: try single-symbol deletions; accept any that keep validity.
    best = word
    best_len = len(word)
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try deletions from left to right.
        for i in range(len(best) - n + 1):
            if time.time() > DEADLINE:
                break
            cand = best[:i] + best[i + 1:]
            if len(cand) < n:
                break
            seen = roll_counts2(cand)
            if is_valid_counts(seen):
                best = cand
                best_len = len(cand)
                improved = True
                break  # restart scan with the new (shorter) word
        # Optional: cheap transposition pass (swap adjacent) if time remains.
        if not improved and time.time() < DEADLINE - 5:
            for i in range(len(best) - 1):
                if time.time() > DEADLINE - 5:
                    break
                cand = best[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                seen = roll_counts2(cand)
                if is_valid_counts(seen):
                    best = cand
                    improved = True
                    break

    # Final safety: ensure the returned word is actually valid; if anything went
    # wrong, fall back to the guaranteed-valid recursive word.
    final = best
    # Cheap full verification (one pass) to be safe before returning.
    present = set()
    h = 0
    for x in final[:n]:
        h = h * base + x
    present.add(h)
    for i in range(n, len(final)):
        h = (h - final[i - n] * powb) * base + final[i]
        present.add(h)
    if present != target_hashes:
        final = _build_recursive(n)

    return "".join(map(str, final))
