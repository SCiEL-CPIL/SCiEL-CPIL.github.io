import math
import random
import time
from fractions import Fraction
from collections import deque

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)

# Precompute reciprocal costs as Fractions
_COSTS = [Fraction(1, p) for p in _PRIMES]

# Precompute coverage masks for faster evaluation
# For each prime index i, and residue r (1..p), store the list of numbers covered in 1..MAX_CHECK
# We'll compute these lazily or precompute for small L
MAX_CHECK = 100000

def _coverage_length(assign):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    MAX_CHECK = 100000
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        # r is 1..p (we use 1-based residues in the assign list for clarity, but 0 means unused)
        # Actually, let's stick to 0..p-1 for residues, but 0 is ambiguous with unused.
        # Let's use -1 for unused or a separate flag.
        # The current code uses 0 for unused and 1..p for residues?
        # "0 <= r < p and 0 means 'not used'" -> This is contradictory if r=0 is a valid residue.
        # The seed solution uses: r = (i + 1) % p; if r == 0: r = 1.
        # So residues are 1..p. 0 means unused.
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)

def _seed_solution():
    """The i-th prime from 13 upward covers the integer i."""
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = 1
        assign[idx] = r
    return assign

def _greedy_construct(target_L, time_limit):
    """
    Greedily add prime/residue pairs to cover 1..target_L.
    Returns assign list.
    """
    start = time.time()
    assign = [0] * NUM_PRIMES
    current_cost = Fraction(0)
    covered = bytearray(target_L + 1)
    
    # Precompute for each prime, the best residue for each "state" of uncovered numbers?
    # Too complex. Just greedy: pick the prime/residue that covers the most uncovered numbers
    # among the first few uncovered numbers, or globally?
    
    # Let's focus on covering the first gap.
    # Find the first uncovered number.
    # Try all unused primes and all residues. Pick the one that covers the most uncovered numbers
    # in the range [1, target_L], with a tie-breaker for covering the first gap.
    
    prime_indices = list(range(NUM_PRIMES))
    
    while True:
        if time.time() - start > time_limit:
            break
        
        # Find first uncovered
        first_uncov = -1
        for n in range(1, target_L + 1):
            if not covered[n]:
                first_uncov = n
                break
        
        if first_uncov == -1:
            break  # All covered
        
        best_prime = -1
        best_residue = -1
        best_score = -1
        
        for i in prime_indices:
            if assign[i] != 0:
                continue
            p = _PRIMES[i]
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            
            # Try all residues 1..p
            for r in range(1, p + 1):
                # Count how many uncovered numbers this covers
                count = 0
                for n in range(r, target_L + 1, p):
                    if not covered[n]:
                        count += 1
                
                # Bonus for covering the first uncovered number
                bonus = 0
                if (first_uncov - r) % p == 0 and first_uncov >= r:
                    bonus = target_L  # Large bonus to prioritize closing the gap
                
                score = count + bonus
                
                if score > best_score:
                    best_score = score
                    best_prime = i
                    best_residue = r
        
        if best_prime == -1:
            break  # No more primes can be added
        
        # Apply
        assign[best_prime] = best_residue
        current_cost += _COSTS[best_prime]
        p = _PRIMES[best_prime]
        r = best_residue
        for n in range(r, target_L + 1, p):
            covered[n] = 1
    
    return assign

def _local_search(assign, time_limit):
    """
    Hill-climbing local search.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try swapping residues for used primes
        used_primes = [i for i in range(NUM_PRIMES) if current[i] != 0]
        
        for i in used_primes:
            if time.time() - start > time_limit:
                return current, current_L
            p = _PRIMES[i]
            old_r = current[i]
            
            # Try each other residue
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                
                new_assign = current[:]
                new_assign[i] = r
                new_L = _coverage_length(new_assign)
                if new_L > current_L:
                    current = new_assign
                    current_L = new_L
                    improved = True
                    break
        
        # Try adding unused primes
        if not improved:
            unused_primes = [i for i in range(NUM_PRIMES) if current[i] == 0]
            for i in unused_primes:
                if time.time() - start > time_limit:
                    return current, current_L
                if current_cost + _COSTS[i] > Fraction(1, 1):
                    continue
                
                p = _PRIMES[i]
                best_r = -1
                best_L = current_L
                
                for r in range(1, p + 1):
                    new_assign = current[:]
                    new_assign[i] = r
                    new_L = _coverage_length(new_assign)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                
                if best_r != -1 and best_L > current_L:
                    current = current[:]
                    current[i] = best_r
                    current_L = best_L
                    current_cost += _COSTS[i]
                    improved = True
        
        # Try removing primes that don't hurt coverage
        if not improved:
            for i in used_primes[:]:
                if time.time() - start > time_limit:
                    return current, current_L
                
                new_assign = current[:]
                new_assign[i] = 0
                new_L = _coverage_length(new_assign)
                if new_L >= current_L:
                    current = new_assign
                    current_cost -= _COSTS[i]
                    improved = True
    
    return current, current_L

def _simulated_annealing(assign, time_limit):
    """
    Simulated annealing.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    best = current[:]
    best_L = current_L
    
    T = 1.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        if T < 0.01:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            used = [i for i in range(NUM_PRIMES) if current[i] != 0]
            if not used:
                continue
            i = random.choice(used)
            p = _PRIMES[i]
            old_r = current[i]
            r = random.randint(1, p)
            if r == old_r:
                continue
            current[i] = r
            new_L = _coverage_length(current)
            delta_L = new_L - current_L
            current_L = new_L
        elif move_type < 0.8:
            unused = [i for i in range(NUM_PRIMES) if current[i] == 0]
            if not unused:
                continue
            i = random.choice(unused)
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            p = _PRIMES[i]
            r = random.randint(1, p)
            current[i] = r
            new_L = _coverage_length(current)
            delta_L = new_L - current_L
            current_L = new_L
            current_cost += _COSTS[i]
        else:
            used = [i for i in range(NUM_PRIMES) if current[i] != 0]
            if not used:
                continue
            i = random.choice(used)
            old_r = current[i]
            current[i] = 0
            new_L = _coverage_length(current)
            delta_L = new_L - current_L
            current_L = new_L
            current_cost -= _COSTS[i]
        
        if delta_L >= 0:
            pass
        else:
            if random.random() < math.exp(delta_L / T):
                pass
            else:
                # Revert
                if move_type < 0.5:
                    current[i] = old_r
                elif move_type < 0.8:
                    current[i] = 0
                    current_cost -= _COSTS[i]
                else:
                    current[i] = old_r
                    current_cost += _COSTS[i]
                current_L = _coverage_length(current)
        
        if current_L > best_L:
            best = current[:]
            best_L = current_L
        
        T *= cooling
    
    return best, best_L

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin
    
    # 1. Start with seed
    best_assign = _seed_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # 2. Try greedy construction for a higher target
    # The seed covers about 900. Let's try to greedily cover up to 1500.
    if time.time() < deadline - 60:
        greedy_assign = _greedy_construct(1500, 30)
        greedy_L = _coverage_length(greedy_assign)
        if greedy_L > best_L and _is_valid(greedy_assign):
            best_assign = greedy_assign
            best_L = greedy_L
    
    # 3. Local search
    if time.time() < deadline - 30:
        ls_assign, ls_L = _local_search(best_assign, min(30, deadline - time.time()))
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # 4. Simulated annealing
    if time.time() < deadline - 10:
        sa_assign, sa_L = _simulated_annealing(best_assign, min(10, deadline - time.time()))
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return _seed_solution()
