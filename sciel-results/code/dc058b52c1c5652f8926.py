"""Stronger solver for Erdos #689 (n=1000) doubly-covered run.

Key insight: for x in 1..L we need at least 2 primes p with x ≡ r_p (mod p).
For large primes (p > L), r_p = 0 means x is covered iff p|x, which only happens
when x >= p > L, so they contribute nothing to 1..L. We can set them to 0.

For small primes (p <= L), we need to assign residues so that every x in 1..L
is hit by at least 2 of the APs.

Strategy:
1. Greedy construction: process x from 1 to L, and for each x ensure it has
   at least 2 covering primes. Use a backtracking/greedy approach.
2. Since this is a covering problem, use a local search / simulated annealing
   approach to find good residue assignments.

The score is the largest L such that every x in 1..L is covered by >= 2 primes.
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 115.0  # Safety margin before 120s
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


def _fast_verify_prefix(residues, small_primes, small_residues, max_check):
    """Check coverage for x in 1..max_check using only small primes.
    Returns the largest L such that all x in 1..L have >= 2 covers."""
    n = len(small_primes)
    if n == 0:
        return 0
    
    # For each small prime, compute which x in 1..max_check it covers
    # Use a difference array approach for each prime
    cover_count = np.zeros(max_check + 1, dtype=np.int8)
    
    for i in range(n):
        p = small_primes[i]
        r = small_residues[i]
        # x ≡ r (mod p) means x = r + kp for k >= 0
        # First x >= 1: if r >= 1, start = r; else start = r + p*ceil((1-r)/p)
        if r >= 1:
            start = r
        else:
            # r = 0, need x = kp >= 1, so start = p
            start = p
        if start > max_check:
            continue
        # Mark all x = start, start+p, start+2p, ... <= max_check
        for x in range(start, max_check + 1, p):
            cover_count[x] += 1
    
    # Find largest L where all 1..L have count >= 2
    for L in range(max_check, 0, -1):
        if cover_count[L] < 2:
            return L - 1
    return max_check


def _construct_greedy_L(target_L):
    """Try to construct residues that cover 1..target_L doubly.
    Uses a greedy assignment for small primes."""
    small = [p for p in _PRIMES if p <= target_L]
    n_small = len(small)
    residues = [0] * _N_PRIMES
    
    # For each x in 1..target_L, track which primes can cover it
    # We need to assign r_p such that each x gets >= 2 primes
    
    # Greedy: for each prime, try to assign a residue that helps cover uncovered numbers
    # This is complex; use a simpler approach:
    # Set r_p = x mod p for some strategic x
    
    # Alternative: use the fact that for x to be covered by p, we need r_p = x mod p
    # So for each x, the set of primes that CAN cover it is all small primes
    # We need to choose r_p values so that the "intersection" gives >= 2 covers
    
    # Simple heuristic: for each small prime p, set r_p to a value that
    # maximizes the number of currently-undercovered x's it would cover
    cover_count = np.zeros(target_L + 1, dtype=np.int8)
    assigned = {}
    
    # Process primes in increasing order
    for p in small:
        # Try all possible residues and pick the one that helps most
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Count how many x in 1..target_L with cover_count[x] < 2 
            # would get an additional cover
            gain = 0
            if r >= 1:
                start = r
            else:
                start = p
            if start > target_L:
                continue
            for x in range(start, target_L + 1, p):
                if cover_count[x] < 2:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        assigned[p] = best_r
        # Update cover counts
        if best_r >= 1:
            start = best_r
        else:
            start = p
        if start <= target_L:
            for x in range(start, target_L + 1, p):
                cover_count[x] += 1
    
    for p, r in assigned.items():
        residues[_BY_P[p]] = r
    
    # Check if this works
    if np.all(cover_count[1:target_L+1] >= 2):
        return residues
    return None


def _simulated_annealing(initial_residues, initial_score, time_budget):
    """Improve residues using simulated annealing."""
    rng = random.Random(int(time.time() * 1000) % (2**31))
    best_res = list(initial_residues)
    best_score = initial_score
    current_res = list(initial_residues)
    current_score = initial_score
    
    # Only modify small primes (p <= current_score + margin)
    end_time = time.time() + time_budget
    T = 50.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < end_time:
        # Pick a prime to modify
        # Focus on primes that are <= best_score + 200
        max_p_for_mod = min(1000, best_score + 200)
        candidates = [i for i, p in enumerate(_PRIMES) if p <= max_p_for_mod]
        if not candidates:
            break
        idx = rng.choice(candidates)
        p = _PRIMES[idx]
        old_r = current_res[idx]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        trial = current_res
        trial[idx] = new_r
        # Quick check: verify up to best_score + 1000
        check_limit = min(MAX_L, best_score + 2000)
        trial_score = _verify(trial, check_limit)
        
        if trial_score >= current_score:
            current_res = trial
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_res = list(trial)
        else:
            # Accept with probability exp((new-old)/T)
            delta = trial_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    current_res = trial
                    current_score = trial_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_score, best_res


def _local_search(residues, score, time_budget):
    """Local search: try changing one residue at a time."""
    rng = random.Random(int(time.time() * 1000) % (2**31) + 1)
    best_res = list(residues)
    best_score = score
    end_time = time.time() + time_budget
    
    while time.time() < end_time:
        improved = False
        max_p_for_mod = min(1000, best_score + 300)
        # Try each prime that could affect the score
        for idx, p in enumerate(_PRIMES):
            if p > max_p_for_mod:
                break
            if time.time() > end_time:
                break
            old_r = best_res[idx]
            # Try a few random residues
            for _ in range(5):
                new_r = rng.randint(0, p - 1)
                if new_r == old_r:
                    continue
                trial = best_res
                trial[idx] = new_r
                check_limit = min(MAX_L, best_score + 1500)
                trial_score = _verify(trial, check_limit)
                if trial_score > best_score:
                    best_score = trial_score
                    best_res = list(trial)
                    improved = True
                    break
        if not improved:
            # No improvement in this round; do a random restart of some residues
            for _ in range(10):
                idx = rng.randint(0, _N_PRIMES - 1)
                p = _PRIMES[idx]
                if p > max_p_for_mod:
                    continue
                best_res[idx] = rng.randint(0, p - 1)
            s = _verify(best_res, min(MAX_L, best_score + 1500))
            if s > best_score:
                best_score = s
    
    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)
    
    best_score = 0
    best_res = [0] * _N_PRIMES
    
    # Heuristic 1: r_p = 0 for all p
    res1 = [0] * _N_PRIMES
    s1 = _verify(res1, 5000)
    if s1 > best_score:
        best_score = s1
        best_res = res1
    
    # Heuristic 2: r_p = 1 for all p
    res2 = [1 % p for p in _PRIMES]
    s2 = _verify(res2, 5000)
    if s2 > best_score:
        best_score = s2
        best_res = res2
    
    # Heuristic 3: Try greedy for various target L
    for target_L in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        if time.time() > DEADLINE - 30:
            break
        res_g = _construct_greedy_L(target_L)
        if res_g is not None:
            s_g = _verify(res_g, target_L + 1000)
            if s_g > best_score:
                best_score = s_g
                best_res = res_g
    
    # Simulated annealing from best
    if time.time() < DEADLINE - 10:
        budget = min(40.0, DEADLINE - time.time() - 10)
        s_sa, r_sa = _simulated_annealing(best_res, best_score, budget)
        if s_sa > best_score:
            best_score = s_sa
            best_res = r_sa
    
    # Local search
    if time.time() < DEADLINE - 5:
        budget = min(20.0, DEADLINE - time.time() - 5)
        s_ls, r_ls = _local_search(best_res, best_score, budget)
        if s_ls > best_score:
            best_score = s_ls
            best_res = r_ls
    
    # Final verification
    final_score = _verify(best_res, MAX_L)
    if final_score > best_score:
        best_score = final_score
    
    return best_res
