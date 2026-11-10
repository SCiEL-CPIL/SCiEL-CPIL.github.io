"""CSP-based solver for Erdos #689 (n=1000) doubly-covered run.

Structural approach: Backtracking constraint satisfaction.
- Process integers x = 1, 2, ..., L in order.
- For each x, identify primes p ≤ L where x is still "compatible" (no prior assignment contradicts).
- Ensure at least 2 primes cover x by assigning their residues.
- Use constraint propagation: when r_p is set, it affects all future x ≡ r_p (mod p).
- For p > L: set r_p = 0 (safe, never hits x in 1..L).
"""
import time
import sys
import numpy as np

T_START = time.time()
DEADLINE = 118.0
MAX_L = 100000
sys.setrecursionlimit(100000)


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
_SMALL_IDX = [i for i, p in enumerate(_PRIMES) if p <= 10000]  # small primes for coverage
_SMALL_PRIMES = [_PRIMES[i] for i in _SMALL_IDX]


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


def _solve_csp(L, seed=0, time_budget=5.0):
    """Backtracking CSP: assign residues for primes p <= L to doubly cover 1..L.
    
    Returns (success, residues_list) or (False, None).
    """
    import random
    rng = random.Random(seed)
    start_time = time.time()
    
    # Only small primes (p <= L) matter for coverage
    small_primes = [p for p in _PRIMES if p <= L]
    small_count = len(small_primes)
    
    if small_count < 2:
        return False, None
    
    # residues[p_index] = assigned residue, -1 if unassigned
    residues = [-1] * _N_PRIMES
    
    # For each small prime, track which residue is assigned
    # For each x in 1..L, track how many primes already cover it
    # and which primes are "forced" to cover it
    
    # Strategy: process x from 1 to L. For each x, find primes p <= L 
    # where x is compatible with current assignments.
    # A prime p is compatible with x if:
    #   - r_p is unassigned, OR r_p == x % p
    # We need at least 2 compatible primes for each x.
    
    # To make this tractable, we use a greedy-with-backtracking approach:
    # Process x in order. For each x, if it already has >= 2 covering primes, skip.
    # Otherwise, we must assign some unassigned primes to cover x.
    # Choose primes that cover the most future integers (greedy heuristic).
    
    # Precompute: for each prime p, the list of x in 1..L that p could cover
    # (i.e., x % p == r_p for some r_p, which is always true for some r_p)
    # Actually, any x can be covered by any p by choosing r_p = x % p.
    
    # The constraint is: once r_p is fixed, p can only cover x where x % p == r_p.
    
    # Greedy with backtracking:
    # For x = 1, 2, ..., L:
    #   count how many primes already cover x (r_p assigned and r_p == x % p)
    #   if count >= 2: continue
    #   if count == 1: need 1 more prime to cover x
    #   if count == 0: need 2 more primes to cover x
    #   Choose unassigned primes to assign r_p = x % p
    #   Prefer primes that cover many future uncovered integers
    
    # For efficiency, we limit backtracking depth and use time budget.
    
    best_solution = None
    
    # Try multiple random orderings of prime selection
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        residues = [-1] * _N_PRIMES
        success = True
        
        # Track coverage count for each x
        cov_count = [0] * (L + 1)
        
        # For each x, which primes cover it (among assigned ones)
        # We'll recompute as needed for efficiency
        
        # Process x from 1 to L
        for x in range(1, L + 1):
            if time.time() - start_time > time_budget:
                success = False
                break
            
            # Count how many assigned primes cover x
            count = 0
            for pi in _SMALL_IDX:
                p = _PRIMES[pi]
                if p > L:
                    break
                r = residues[pi]
                if r >= 0 and x % p == r:
                    count += 1
            
            if count >= 2:
                continue
            
            need = 2 - count
            
            # Find unassigned small primes that can cover x
            # (any unassigned prime p <= L can cover x by setting r_p = x % p)
            candidates = []
            for pi in _SMALL_IDX:
                p = _PRIMES[pi]
                if p > L:
                    break
                if residues[pi] < 0:
                    candidates.append(pi)
            
            if len(candidates) < need:
                success = False
                break
            
            # Heuristic: prefer primes that cover many future integers
            # that are currently under-covered
            # For simplicity, prefer smaller primes (they cover more frequently)
            # Actually, smaller p means x % p cycles faster, so p covers ~L/p integers
            # We want to maximize future coverage, so prefer primes that cover
            # integers with low current coverage
            
            # Simple heuristic: pick the smallest 'need' unassigned primes
            # that are <= L
            candidates.sort(key=lambda pi: _PRIMES[pi])
            
            # But we need to be careful: assigning r_p = x % p for a small p
            # means p will cover x, x+p, x+2p, etc.
            # This might conflict with future requirements.
            
            # For now, greedy: pick the smallest 'need' unassigned primes
            chosen = candidates[:need]
            
            for pi in chosen:
                p = _PRIMES[pi]
                residues[pi] = x % p
                cov_count[x] += 1
        
        if success:
            # Verify
            full_res = [0] * _N_PRIMES
            for pi in range(_N_PRIMES):
                if residues[pi] >= 0:
                    full_res[pi] = residues[pi]
                else:
                    full_res[pi] = 0  # safe for p > L
            
            score = _verify(full_res, L + 100)
            if score >= L:
                best_solution = full_res
                break
    
    if best_solution:
        return True, best_solution
    return False, None


def _hill_climb(residues, base_score, budget=2.0):
    """Try to improve by adjusting some residues."""
    best_score = base_score
    best_res = list(residues)
    end_time = time.time() + budget
    rng = random.Random(int(time.time() * 1000))
    
    while time.time() < end_time:
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


import random


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)
    
    # Start with the seed solution
    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]
    
    # Try CSP solver for increasing L
    # The CSP is hard, so start with modest L and increase
    for target_L in [100, 200, 500, 1000, 2000, 5000]:
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple seeds for the CSP
        for seed in range(5):
            if time.time() > DEADLINE - 10:
                break
            
            success, res = _solve_csp(target_L, seed=seed, time_budget=3.0)
            if success and res:
                score = _verify(res, target_L + 5000)
                if score > best_score:
                    best_score = score
                    best_res = res
                    # Hill climb from here
                    if time.time() < DEADLINE - 5:
                        s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(3.0, DEADLINE - time.time() - 5))
                        if s_hc > best_score:
                            best_score = s_hc
                            best_res = r_hc
                break  # Found a solution for this L, try next L
    
    # Also try the simple heuristics as fallback
    if time.time() < DEADLINE - 5:
        res1 = [0] * _N_PRIMES
        s1 = _verify(res1, 5000)
        if s1 > best_score:
            best_score = s1
            best_res = res1
        
        res2 = [1 % p for p in _PRIMES]
        s2 = _verify(res2, 5000)
        if s2 > best_score:
            best_score = s2
            best_res = res2
    
    # Final hill climb
    if time.time() < DEADLINE - 3:
        s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(5.0, DEADLINE - time.time() - 3))
        if s_hc > best_score:
            best_score = s_hc
            best_res = r_hc
    
    return best_res
