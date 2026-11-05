"""Construct the largest admissible tuple within [0, 5000]."""
import time
from collections import defaultdict

N = 5000
DEADLINE = 325  # safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _construct(primes, deadline):
    """
    Greedy removal with overlap-aware class choice.

    Start with all points [0, N]. For each prime p (increasing order), if all
    p residue classes are still fully present, remove the class that currently
    has the FEWEST surviving points (i.e. the class most already destroyed by
    earlier deletions), so deletions overlap and total removed is minimized.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    # surviving count per class for each prime processed so far, lazily
    # we recompute counts each prime; N is only 5000 so this is cheap.

    for p in primes:
        if p > N:
            break
        if time.time() - start > deadline:
            break

        # count surviving points per residue class mod p
        counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                counts[v % p] += 1

        # if some class already empty, constraint satisfied
        if any(c == 0 for c in counts):
            continue

        # choose class with minimum surviving count (max overlap)
        min_count = min(counts)
        # tie-break: prefer smaller class index for determinism
        target = counts.index(min_count)
        for v in range(target, N + 1, p):
            present[v] = 0

    return [v for v in range(N + 1) if present[v]]


def _refine(result, primes, deadline):
    """
    Local search: for each prime p <= len(result) that is currently fully covered,
    try removing each candidate class and keep the one that removes the fewest
    points while preserving admissibility for all smaller primes. This is a
    cautious improvement; we only apply if it strictly increases size.
    """
    start = time.time()
    result_set = set(result)

    for p in primes:
        if p > len(result_set):
            break
        if time.time() - start > deadline:
            break
        residues = set(v % p for v in result_set)
        if len(residues) < p:
            continue  # already satisfied
        # currently fully covered; we must remove a class.
        # find the class whose removal loses the fewest points
        class_members = defaultdict(list)
        for v in result_set:
            class_members[v % p].append(v)
        best_class = min(class_members, key=lambda c: len(class_members[c]))
        removed = class_members[best_class]
        result_set = result_set - set(removed)
        result = sorted(result_set)

    return result


def construct():
    start = time.time()
    deadline = start + DEADLINE
    primes = _primes(N)

    result = _construct(primes, deadline - (time.time() - start))

    # verify; if invalid, force-fix by removing classes
    if not _is_admissible(result, primes):
        result_set = set(result)
        for p in primes:
            if p > len(result_set):
                break
            residues = set(v % p for v in result_set)
            if len(residues) == p:
                class_members = defaultdict(list)
                for v in result_set:
                    class_members[v % p].append(v)
                best = min(class_members, key=lambda c: len(class_members[c]))
                result_set -= set(class_members[best])
        result = sorted(result_set)

    return sorted(set(result))


# EVOLVE-BLOCK-START
# Entry point is construct(); logic above.
# EVOLVE-BLOCK-END
