# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

def _score(residues, primes, cap=100000):
    """Return the largest L such that every integer 1..L has count >= 2."""
    # Build an array of counts for 1..cap
    # For each prime p with residue r, it covers r, r+p, r+2p, ... (>= 1)
    counts = np.zeros(cap + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        # Start at max(r, 1), then r+p, r+2p, ...
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= cap:
            counts[start::p] += 1
    # Find the largest L where min(counts[1..L]) >= 2
    # Find first position where count < 2
    below = np.where(counts[1:cap+1] < 2)[0]
    if len(below) == 0:
        return cap
    return int(below[0])

def construct():
    deadline = time.time() + 140  # safety margin
    
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    
    # Initial assignment: use a structured approach
    # Small primes (p <= 21): assign to cover small integers densely
    # Large primes: random
    best_residues = None
    best_score = 0
    
    # Strategy 1: Structured initial + greedy repair
    # Assign small primes to cover integers 1..n/2 densely
    # For prime i (0-indexed), assign residue to cover integer (i//2 + 1)
    residues = [0] * n
    for i, p in enumerate(primes):
        # Each integer gets covered by ~2 primes
        target = (i // 2) + 1
        residues[i] = target % p
    
    # Greedy repair: find uncovered integers, fix them
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        sc = _score(residues, primes, cap=2000)
        if sc > best_score:
            best_score = sc
            best_residues = residues[:]
        if sc <= 0:
            break
        # Find first integer with count < 2
        # Check a few positions
        check_L = min(sc + 100, 500)
        counts = np.zeros(check_L + 1, dtype=np.int32)
        for p, r in zip(primes, residues):
            start = r
            if start < 1:
                start += ((1 - start + p - 1) // p) * p
            if start <= check_L:
                counts[start::p] += 1
        below = np.where(counts[1:check_L+1] < 2)[0]
        if len(below) == 0:
            continue
        bad_x = int(below[0]) + 1  # 1-indexed
        # Find primes that currently DON'T cover bad_x, and whose residue we can change
        # Changing residue of prime p to (bad_x % p) makes it cover bad_x
        # But we need to make sure we don't break coverage elsewhere
        # Pick a large prime (less impact)
        candidates = [i for i in range(n) if primes[i] > 100]
        random.shuffle(candidates)
        improved = False
        for idx in candidates[:20]:
            old_r = residues[idx]
            new_r = bad_x % primes[idx]
            if old_r == new_r:
                continue
            residues[idx] = new_r
            new_sc = _score(residues, primes, cap=2000)
            if new_sc > sc:
                improved = True
                break
            else:
                residues[idx] = old_r
        if not improved:
            # Try changing a small prime's residue to cover bad_x
            small_candidates = [i for i in range(n) if primes[i] <= 100]
            random.shuffle(small_candidates)
            for idx in small_candidates[:10]:
                old_r = residues[idx]
                new_r = bad_x % primes[idx]
                if old_r == new_r:
                    continue
                residues[idx] = new_r
                new_sc = _score(residues, primes, cap=2000)
                if new_sc > sc:
                    improved = True
                    break
                else:
                    residues[idx] = old_r
    
    # Local search: random perturbations
    if best_residues is not None:
        residues = best_residues[:]
        sc = _score(residues, primes, cap=5000)
        if sc > best_score:
            best_score = sc
            best_residues = residues[:]
    
    for it in range(5000):
        if time.time() > deadline - 5:
            break
        if best_residues is None:
            break
        residues = best_residues[:]
        # Randomly change 1-3 residues
        k = random.randint(1, 3)
        for _ in range(k):
            idx = random.randint(0, n - 1)
            old_r = residues[idx]
            residues[idx] = random.randint(0, primes[idx] - 1)
        sc = _score(residues, primes, cap=5000)
        if sc > best_score:
            best_score = sc
            best_residues = residues[:]
    
    # Final: try to extend using a more careful repair
    if best_residues is not None:
        residues = best_residues[:]
        # Multi-pass greedy repair with larger cap
        for pass_num in range(50):
            if time.time() > deadline - 2:
                break
            sc = _score(residues, primes, cap=10000)
            if sc > best_score:
                best_score = sc
                best_residues = residues[:]
            if sc >= 1000:
                break
            # Find first gap
            check_L = min(sc + 200, 10000)
            counts = np.zeros(check_L + 1, dtype=np.int32)
            for p, r in zip(primes, residues):
                start = r
                if start < 1:
                    start += ((1 - start + p - 1) // p) * p
                if start <= check_L:
                    counts[start::p] += 1
            below = np.where(counts[1:check_L+1] < 2)[0]
            if len(below) == 0:
                continue
            bad_x = int(below[0]) + 1
            # Try to fix by changing one residue
            fixed = False
            # Prefer larger primes for less disruption
            order = sorted(range(n), key=lambda i: -primes[i])
            for idx in order:
                old_r = residues[idx]
                new_r = bad_x % primes[idx]
                if old_r == new_r:
                    continue
                residues[idx] = new_r
                new_sc = _score(residues, primes, cap=10000)
                if new_sc >= sc:
                    fixed = True
                    break
                else:
                    residues[idx] = old_r
            if not fixed:
                # Try small primes
                for idx in range(min(30, n)):
                    old_r = residues[idx]
                    new_r = bad_x % primes[idx]
                    if old_r == new_r:
                        continue
                    residues[idx] = new_r
                    new_sc = _score(residues, primes, cap=10000)
                    if new_sc >= sc:
                        fixed = True
                        break
                    else:
                        residues[idx] = old_r
            if not fixed:
                break
    
    if best_residues is not None:
        return best_residues
    # Fallback: seed-style
    return [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
