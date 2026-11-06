"""Stronger program: greedy set-cover heuristic + local polish for double covering."""
# EVOLVE-BLOCK-START
import time

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _score(residues, primes):
    maxL = 100000
    counts = [0] * (maxL + 1)
    for r, p in zip(residues, primes):
        start = p if r == 0 else r
        for x in range(start, maxL + 1, p):
            counts[x] += 1
    L = 0
    for x in range(1, maxL + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin under 120s
    primes = _primes_upto(1000)
    n = len(primes)
    maxL = 100000

    # coverage counts
    counts = [0] * (maxL + 1)
    residues = [0] * n

    # Helper: benefit of adding residue r to prime p = # positions with count==1
    # that lie in the arithmetic progression.
    def benefit(r, p):
        start = p if r == 0 else r
        b = 0
        for x in range(start, maxL + 1, p):
            if counts[x] == 1:
                b += 1
        return b

    # --- Phase 1: Greedy construction (increasing prime order) ---
    for idx in range(n):
        p = primes[idx]
        best_r = 0
        best_b = -1
        for r in range(p):
            b = benefit(r, p)
            if b > best_b:
                best_b = b
                best_r = r
        residues[idx] = best_r
        # apply
        start = p if best_r == 0 else best_r
        for x in range(start, maxL + 1, p):
            counts[x] += 1

    best = _score(residues, primes)

    # --- Phase 2: Local polish (single-residue hill climbing, a few passes) ---
    max_passes = 30
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            p = primes[idx]
            cur = residues[idx]
            # remove
            start = p if cur == 0 else cur
            for x in range(start, maxL + 1, p):
                counts[x] -= 1
            best_r = cur
            best_b = benefit(cur, p)
            for r in range(p):
                if r == cur:
                    continue
                b = benefit(r, p)
                if b > best_b:
                    best_b = b
                    best_r = r
            # apply
            start = p if best_r == 0 else best_r
            for x in range(start, maxL + 1, p):
                counts[x] += 1
            if best_r != cur:
                residues[idx] = best_r
                improved = True
        curL = _score(residues, primes)
        if curL > best:
            best = curL
        if not improved:
            break

    return residues
# EVOLVE-BLOCK-END
