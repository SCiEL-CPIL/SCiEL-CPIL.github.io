import time
import random
import math
import sys
import numpy as np
from functools import lru_cache

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)
PRIME_SET = set(PRIMES)

def _compute_survivors(assignments):
    """Compute the set of survivors given residue assignments using numpy for speed."""
    # Start with all numbers
    survivors = np.ones(N + 1, dtype=bool)
    
    for p, r in assignments.items():
        if p <= N:
            # Create mask: x % p != r
            mask = np.arange(N + 1) % p != r
            survivors &= mask
    
    return survivors

def _get_violating_prime(survivors):
    """Find a prime p <= len(survivors) where all residues mod p are present."""
    count = survivors.sum()
    if count == 0:
        return None
    
    for p in PRIMES:
        if p > count:
            break
        # Check if all residues mod p are present
        residues = np.zeros(p, dtype=int)
        idx = np.where(survivors)[0]
        if len(idx) == 0:
            break
        np.add.at(residues, idx % p, 1)
        if np.all(residues > 0):
            return p
    return None

def _greedy_construct(assignments, rng=None):
    """Complete assignments greedily to get an admissible set."""
    survivors = _compute_survivors(assignments)
    committed = dict(assignments)
    
    while True:
        count = survivors.sum()
        if count == 0:
            break
        
        viol_p = _get_violating_prime(survivors)
        if viol_p is None:
            break
        
        p = viol_p
        # Find residue with minimum count
        idx = np.where(survivors)[0]
        if len(idx) == 0:
            break
        counts = np.bincount(idx % p, minlength=p)
        min_count = counts.min()
        candidates = np.where(counts == min_count)[0]
        
        if rng is not None:
            r_to_remove = int(rng.choice(candidates))
        else:
            r_to_remove = int(candidates[0])
        
        committed[p] = r_to_remove
        survivors &= (np.arange(N + 1) % p != r_to_remove)
    
    return survivors, committed

def _local_search_improve(survivors, committed, max_iterations=1000, rng=None):
    """Try to improve the solution by changing residue assignments."""
    if rng is None:
        rng = random.Random(42)
    
    best_survivors = survivors.copy()
    best_committed = dict(committed)
    best_count = best_survivors.sum()
    
    for _ in range(max_iterations):
        # Pick a random prime from committed or try adding a new one
        if committed and rng.random() < 0.7:
            # Change an existing assignment
            p = rng.choice(list(committed.keys()))
            old_r = committed[p]
            new_r = rng.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Try the change
            test_committed = dict(committed)
            test_committed[p] = new_r
            test_survivors = _compute_survivors(test_committed)
            
            # Complete greedily
            test_survivors, test_committed = _greedy_construct(test_committed, rng)
            test_count = test_survivors.sum()
            
            if test_count > best_count:
                best_survivors = test_survivors
                best_committed = test_committed
                best_count = test_count
        else:
            # Try adding a new prime assignment
            # Find primes not in committed that are <= current size
            current_size = best_survivors.sum()
            candidates = [p for p in PRIMES if p <= current_size and p not in committed]
            if not candidates:
                continue
            p = rng.choice(candidates)
            r = rng.randint(0, p - 1)
            
            test_committed = dict(committed)
            test_committed[p] = r
            test_survivors, test_committed = _greedy_construct(test_committed, rng)
            test_count = test_survivors.sum()
            
            if test_count > best_count:
                best_survivors = test_survivors
                best_committed = test_committed
                best_count = test_count
    
    return best_survivors, best_committed

def _simulated_annealing(survivors, committed, start_temp=1.0, end_temp=0.01, 
                          max_iterations=5000, rng=None):
    """Simulated annealing to escape local optima."""
    if rng is None:
        rng = random.Random(42)
    
    best_survivors = survivors.copy()
    best_committed = dict(committed)
    best_count = best_survivors.sum()
    
    current_survivors = survivors.copy()
    current_committed = dict(committed)
    current_count = current_survivors.sum()
    
    for i in range(max_iterations):
        temp = start_temp * (end_temp / start_temp) ** (i / max_iterations)
        
        # Generate neighbor
        if current_committed and rng.random() < 0.8:
            p = rng.choice(list(current_committed.keys()))
            new_r = rng.randint(0, p - 1)
            test_committed = dict(current_committed)
            test_committed[p] = new_r
        else:
            current_size = current_survivors.sum()
            candidates = [p for p in PRIMES if p <= current_size and p not in current_committed]
            if not candidates:
                continue
            p = rng.choice(candidates)
            new_r = rng.randint(0, p - 1)
            test_committed = dict(current_committed)
            test_committed[p] = new_r
        
        test_survivors, test_committed = _greedy_construct(test_committed, rng)
        test_count = test_survivors.sum()
        
        delta = test_count - current_count
        
        if delta > 0 or rng.random() < math.exp(delta / temp):
            current_survivors = test_survivors
            current_committed = test_committed
            current_count = test_count
            
            if current_count > best_count:
                best_survivors = current_survivors.copy()
                best_committed = dict(current_committed)
                best_count = current_count
    
    return best_survivors, best_committed

def _construct():
    start_time = time.time()
    best_survivors = None
    best_count = 0
    best_committed = {}
    
    rng = random.Random(42)
    
    # Phase 1: Try different initial assignments for small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Try all combinations for first 3 primes
    for r2 in [0, 1]:
        for r3 in [0, 1, 2]:
            for r5 in [0, 1, 2, 3, 4]:
                if time.time() - start_time > MAX_TIME - 60:
                    break
                
                assignments = {2: r2, 3: r3, 5: r5}
                survivors, committed = _greedy_construct(assignments, rng)
                count = survivors.sum()
                
                if count > best_count:
                    best_survivors = survivors.copy()
                    best_count = count
                    best_committed = dict(committed)
    
    # Phase 2: Local search improvement
    if time.time() - start_time < MAX_TIME - 30 and best_survivors is not None:
        improved_survivors, improved_committed = _local_search_improve(
            best_survivors, best_committed, max_iterations=200, rng=rng
        )
        if improved_survivors.sum() > best_count:
            best_survivors = improved_survivors
            best_count = improved_survivors.sum()
            best_committed = improved_committed
    
    # Phase 3: Simulated annealing for more thorough search
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 20 and best_survivors is not None:
        max_iter = min(3000, int(time_left * 50))
        sa_survivors, sa_committed = _simulated_annealing(
            best_survivors, best_committed, 
            start_temp=0.5, end_temp=0.01,
            max_iterations=max_iter, rng=rng
        )
        if sa_survivors.sum() > best_count:
            best_survivors = sa_survivors
            best_count = sa_survivors.sum()
            best_committed = sa_committed
    
    # Phase 4: Final local search polish
    if time.time() - start_time < MAX_TIME - 5 and best_survivors is not None:
        final_survivors, final_committed = _local_search_improve(
            best_survivors, best_committed, max_iterations=100, rng=rng
        )
        if final_survivors.sum() > best_count:
            best_survivors = final_survivors
            best_count = final_survivors.sum()
            best_committed = final_committed
    
    # Convert to list
    if best_survivors is not None:
        result = [int(x) for x in np.where(best_survivors)[0]]
    else:
        result = []
    
    return result

def construct():
    return _construct()

if __name__ == "__main__":
    result = construct()
    print(f"Best size: {len(result)}")
