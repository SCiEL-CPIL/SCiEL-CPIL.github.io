"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _score(nums):
    """Count distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            primes |= _prime_factors(s)
    return len(primes)

def _score_with_primes(nums):
    """Count distinct prime factors across all pairwise sums, return (count, set)."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            primes |= _prime_factors(s)
    return len(primes), primes

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best = list(range(1, 13))
    best_score = _score(best)
    
    # Try structured families
    # Family 1: arithmetic progressions
    for a in range(1, 50):
        for d in range(1, 50):
            if time.time() > deadline:
                break
            nums = [a + i * d for i in range(12)]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return best
    
    # Family 2: multiples of a base with various multiplier sets
    for base in range(2, 30):
        if time.time() > deadline:
            break
        # Try several multiplier patterns
        multiplier_sets = [
            list(range(1, 13)),
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        ]
        for mults in multiplier_sets:
            if time.time() > deadline:
                break
            nums = [m * base for m in mults]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            if len(set(nums)) < 12:
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Family 3: numbers of form 2^a * 3^b
    candidates = set()
    for a in range(0, 12):
        for b in range(0, 12):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= 10000:
                candidates.add(val)
    candidates = sorted(candidates)
    if len(candidates) >= 12:
        # Try various subsets
        for start_idx in range(0, max(0, len(candidates) - 12)):
            if time.time() > deadline:
                break
            nums = candidates[start_idx:start_idx + 12]
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Family 4: numbers all divisible by a common factor
    for g in range(2, 20):
        if time.time() > deadline:
            break
        for offset in range(0, 5):
            if time.time() > deadline:
                break
            nums = [g * (i + offset + 1) for i in range(12)]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Local search with multiple restarts
    random.seed(42)
    
    # Try several different initial configurations
    initial_configs = [
        best[:],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
    ]
    
    for config in initial_configs:
        if time.time() > deadline - 15:
            break
        current = config[:]
        if len(set(current)) < 12:
            continue
        current_score = _score(current)
        
        no_improve = 0
        max_no_improve = 150
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            i = random.randint(0, 11)
            old_val = current[i]
            
            # Try several candidate replacements
            candidates = []
            # Small perturbations
            for delta in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]:
                new_val = old_val + delta
                if 1 <= new_val <= 10000 and new_val not in current:
                    candidates.append(new_val)
            # Random values
            for _ in range(5):
                new_val = random.randint(1, 10000)
                if new_val not in current:
                    candidates.append(new_val)
            
            found_improvement = False
            for new_val in candidates:
                if new_val in current:
                    continue
                current[i] = new_val
                sc = _score(current)
                if sc < current_score:
                    current_score = sc
                    no_improve = 0
                    found_improvement = True
                    break
                else:
                    current[i] = old_val
            
            if not found_improvement:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
        
        if current_score < best_score:
            best_score = current_score
            best = current[:]
    
    # Final polish: try targeted improvements on the best solution
    if time.time() < deadline - 5 and best_score > 3:
        current = best[:]
        current_score = best_score
        current_primes, _ = _score_with_primes(current)
        
        no_improve = 0
        max_no_improve = 100
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            i = random.randint(0, 11)
            old_val = current[i]
            
            # Try to find values that make sums have fewer new prime factors
            best_new_val = None
            best_new_score = current_score
            
            for new_val in range(1, 101):
                if new_val in current:
                    continue
                current[i] = new_val
                sc = _score(current)
                if sc < best_new_score:
                    best_new_score = sc
                    best_new_val = new_val
                current[i] = old_val
            
            if best_new_val is not None:
                current[i] = best_new_val
                current_score = best_new_score
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
        
        if current_score < best_score:
            best_score = current_score
            best = current[:]
    
    return best
# EVOLVE-BLOCK-END
