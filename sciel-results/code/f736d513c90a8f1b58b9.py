import time

N = 5000

def _primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N once
_PRIMES = _primes(N)

def construct():
    start = time.time()
    DEADLINE = start + 280.0  # safety margin under 300s

    # Start with all integers in [0, N]
    points = list(range(N + 1))

    # We need admissibility for all primes p <= len(points).
    # As we remove elements, len shrinks. Process primes in ascending order.
    # For each prime p <= current len, ensure not all residues mod p are present.
    # If all present, remove the least-populated residue class.

    # To be safe, iterate primes ascending; recheck len each time.
    for p in _PRIMES:
        if time.time() > DEADLINE:
            break
        L = len(points)
        if p > L:
            break
        # Count residues mod p
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        if min(counts) > 0:
            # All residues present -> remove least-populated class
            # Find the residue with minimum count (tie-break: smallest residue for determinism)
            rmin = min(range(p), key=lambda r: counts[r])
            points = [v for v in points if v % p != rmin]

    # Final safety: verify admissibility and repair if needed (cheap)
    L = len(points)
    for p in _PRIMES:
        if p > L:
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        if min(counts) > 0:
            rmin = min(range(p), key=lambda r: counts[r])
            points = [v for v in points if v % p != rmin]
            L = len(points)

    return points
