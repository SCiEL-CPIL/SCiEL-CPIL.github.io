"""Optimized solver for Erdos #689 (n=1000) doubly-covered run.

Strategy:
  - For primes p > L: set r_p = 0 (safe, no x in 1..L is divisible by p)
  - For primes p <= L: use greedy assignment to maximize doubly-covered prefix
  - Verify and hill-climb to improve
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 118.0
MAX_L = 100000


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)
_BY_P = {p: i for i, p in enumerate(_PRIMES)}
_N_PRIMES = len(_PRIMES)


def _verify(residues, limit=MAX_L):
    """Find largest L such that every x in 1..L has >= 2 covering primes."""
    p = np.asarray(_PRIMES, dtype=np.int64)
    r = np.asarray(residues, dtype=np.int64)
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(p, r):
        rr = int(rr)
        start = rr
        if start < 1:
            start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    ok = mark[1:] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _construct_greedy(L, seed=0):
    """Build residues: large primes get r=0, small primes use two-smallest-divisor scheme."""
    rng = random.Random(seed)
    primes = _PRIMES
    residues = [0] * _N_PRIMES
    
    # Large primes (p > L): r_p = 0 is safe
    # Small primes: assign via two-smallest-divisor scheme
    small = [p for p in primes if p <= L]
    assigned = {}
    ok = True
    for x in range(1, L + 1):
        divs = []
        m = x
        for p in small:
            if p * p > m:
                break
            if m % p == 0:
                divs.append(p)
                while m % p == 0:
                    m //= p
                if len(divs) == 2:
                    break
        if m > 1:
            divs.append(m)
        if len(divs) < 2:
            ok = False
            break
        d1, d2 = divs[0], divs[1]
        for d in (d1, d2):
            rr = x % d
            if d in assigned and assigned[d] != rr:
                ok = False
                break
            assigned[d] = rr
        if not ok:
            break
    if not ok:
        return None
    
    for d, rr in assigned.items():
        residues[_BY_P[d]] = rr
    # Unassigned small primes get r=0
    for p in small:
        if p not in assigned:
            residues[_BY_P[p]] = 0
    # Large primes already 0
    
    return residues


def _hill_climb(residues, base_score, budget=2.0):
    """Try to improve by adjusting some residues."""
    best_score = base_score
    best_res = list(residues)
    end_time = time.time() + budget
    rng = random.Random(int(time.time() * 1000))
    
    while time.time() < end_time:
        # Pick a random prime and try a different residue
        idx = rng.randint(0, _N_PRIMES - 1)
        p = _PRIMES[idx]
        old_r = best_res[idx]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        trial = list(best_res)
        trial[idx] = new_r
        s = _verify(trial, base_score + 2000)
        if s > best_score:
            best_score = s
            best_res = trial
    
    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)
    
    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]
    
    # Try various target L values
    # The two-smallest-divisor scheme works up to L where all x in 1..L have >= 2 prime factors
    # (counting multiplicity doesn't help since we need distinct primes)
    # x must have at least 2 distinct prime divisors, so x >= 6 (2*3)
    # Actually x=4=2^2 has only one distinct prime, so it fails
    # x=6=2*3 works, x=8=2^3 fails, x=9=3^2 fails, x=10=2*5 works
    # So we need all x in 1..L to have >= 2 distinct prime divisors
    # The smallest such x is 6, so L can't exceed... wait, we need EVERY x in 1..L
    # x=1,2,3,4,5 all have < 2 distinct prime divisors
    # So the two-smallest-divisor scheme CANNOT cover 1..L for any L >= 5!
    
    # This means the current approach is fundamentally limited.
    # We need a different strategy: assign residues so that the APs overlap to cover 1..L
    
    # New strategy: for small primes, assign r_p such that the APs are dense
    # Use r_p = 0 for all small primes: then x is covered by p iff p|x
    # Number of prime divisors of x in 1..L varies
    
    # Better: use a covering design approach
    # For each x in 1..L, we need at least 2 primes p with x ≡ r_p (mod p)
    
    # Greedy: process x from 1 to L, ensure coverage
    # For simplicity, try a few heuristic assignments
    
    # Heuristic 1: r_p = 0 for all p (covers multiples of p)
    res1 = [0] * _N_PRIMES
    s1 = _verify(res1, 5000)
    if s1 > best_score:
        best_score = s1
        best_res = res1
    
    # Heuristic 2: r_p = 1 for all p (covers x ≡ 1 mod p)
    res2 = [1] * _N_PRIMES
    # But r_p must be < p, so for p=2, r=1 is ok, for p=3, r=1 ok, etc.
    res2 = [1 % p for p in _PRIMES]
    s2 = _verify(res2, 5000)
    if s2 > best_score:
        best_score = s2
        best_res = res2
    
    # Heuristic 3: alternating 0 and 1
    res3 = [i % 2 for i in range(_N_PRIMES)]
    s3 = _verify(res3, 5000)
    if s3 > best_score:
        best_score = s3
        best_res = res3
    
    # Now try the two-smallest-divisor for L where it works (only for x with >=2 distinct primes)
    # Since 1..5 fail, this scheme can't give L > 5. Skip.
    
    # Try hill climbing from best heuristic
    if time.time() < DEADLINE - 5:
        s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(10.0, DEADLINE - time.time() - 5))
        if s_hc > best_score:
            best_score = s_hc
            best_res = r_hc
    
    # Try more random restarts
    attempts = 0
    while time.time() < DEADLINE - 3 and attempts < 20:
        attempts += 1
        # Random residues
        rand_res = [rng.randint(0, p - 1) for p in _PRIMES]
        s_rand = _verify(rand_res, 5000)
        if s_rand > best_score:
            best_score = s_rand
            best_res = rand_res
            # Hill climb from here
            if time.time() < DEADLINE - 5:
                s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(5.0, DEADLINE - time.time() - 5))
                if s_hc > best_score:
                    best_score = s_hc
                    best_res = r_hc
    
    return best_res
