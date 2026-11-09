"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums(nums):
    """Generate all non-empty subset sums as a set."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sums

def _check_validity_fast(nums):
    """Fast check: all subset sums distinct and no divisibility."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        # Too many sums to enumerate; use probabilistic check
        return False
    
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    
    for s in sums_list:
        if s == 0:
            continue
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    return _check_validity_fast(nums)

def _greedy_build(candidate_pool, target_size, rng, time_limit=2.0):
    """Greedy build: add candidates that maintain validity."""
    start = time.time()
    current = []
    pool = list(candidate_pool)
    rng.shuffle(pool)
    
    for num in pool:
        if time.time() - start > time_limit:
            break
        if len(current) >= target_size:
            break
        trial = current + [num]
        if len(trial) <= 20 and _check_validity_fast(trial):
            current = trial
    
    return current

def _local_search_improve(current, all_candidates, max_steps=500, time_limit=10.0):
    """Try to improve by adding elements or swapping."""
    start = time.time()
    best = list(current)
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        
        # Try adding a new element
        for num in all_candidates:
            if num in best:
                continue
            trial = best + [num]
            if len(trial) <= 20 and _check_validity_fast(trial):
                best = trial
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping
        for i in range(len(best)):
            for num in all_candidates:
                if num in best:
                    continue
                trial = best[:]
                trial[i] = num
                if _check_validity_fast(trial):
                    best = trial
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Known good baseline
    best = [31, 32, 34]
    
    # Candidate pool: focus on numbers in ranges that reduce divisibility risk
    # Larger numbers tend to have fewer divisors in the sum range
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Try different candidate ranges with greedy + local search
    ranges_to_try = [
        list(range(128, 257)),  # Upper half
        list(range(96, 257)),
        list(range(64, 257)),
        list(range(48, 257)),
        list(range(32, 257)),
        all_candidates,
    ]
    
    for cand_range in ranges_to_try:
        if time.time() > deadline - 15:
            break
        
        # Multiple random trials
        for trial_seed in range(20):
            if time.time() > deadline - 15:
                break
            
            rng = random.Random(trial_seed * 104729 + len(cand_range))
            current = _greedy_build(cand_range, target_size=15, rng=rng, time_limit=1.5)
            
            if len(current) > len(best):
                best = current[:]
            
            # Local search to extend
            if time.time() > deadline - 10:
                break
            improved = _local_search_improve(current, all_candidates, max_steps=200, time_limit=5.0)
            if len(improved) > len(best):
                best = improved[:]
    
    # Strategy 2: Systematic search for larger sets
    # Try to build sets of size 10-15 using backtracking with pruning
    if time.time() < deadline - 20:
        # Use a more structured approach: start from known good small sets and grow
        base_sets = [
            [31, 32, 34],
            [63, 64, 66],
            [127, 128, 130],
            [97, 98, 100],
            [191, 192, 194],
        ]
        
        for base in base_sets:
            if time.time() > deadline - 15:
                break
            
            current = list(base)
            # Sort candidates by descending value to prioritize larger numbers
            for num in sorted(all_candidates, reverse=True):
                if num in current:
                    continue
                trial = current + [num]
                if len(trial) <= 18 and _check_validity_fast(trial):
                    current = trial
            
            if len(current) > len(best):
                best = current[:]
            
            # Local search
            if time.time() > deadline - 10:
                break
            improved = _local_search_improve(current, all_candidates, max_steps=300, time_limit=8.0)
            if len(improved) > len(best):
                best = improved[:]
    
    # Strategy 3: Randomized restarts with different heuristics
    if time.time() < deadline - 10:
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(restart * 31337)
            
            # Random initial seed
            seed_size = rng.randint(2, 5)
            initial = rng.sample(all_candidates, seed_size)
            
            current = _greedy_build(all_candidates, target_size=15, rng=rng, time_limit=1.0)
            if len(current) < seed_size:
                current = list(initial)
            
            improved = _local_search_improve(current, all_candidates, max_steps=300, time_limit=3.0)
            if len(improved) > len(best):
                best = improved[:]
    
    # Final local search on best
    if time.time() < deadline - 5:
        final_improved = _local_search_improve(best, all_candidates, max_steps=500, time_limit=10.0)
        if len(final_improved) > len(best):
            best = final_improved[:]
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
