"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _max_sidon_size(ambient):
    """Compute the largest subset of ambient with all unordered pair sums unique (including diagonals)."""
    n = len(ambient)
    if n <= 1:
        return n
    
    # Precompute pair sums for all pairs (i,j) with i<=j
    # A subset S is Sidon if all sums a+b for a,b in S with a<=b are distinct
    # Use bitmask DP or brute force for small n
    
    best = 0
    # Try all subsets by size, from largest to smallest
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(range(n), size):
            # Check if this subset is Sidon
            sums = set()
            is_sidon = True
            for i in range(len(combo)):
                for j in range(i, len(combo)):
                    s = ambient[combo[i]] + ambient[combo[j]]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return size
    return 0

def _max_sidon_size_fast(ambient):
    """Faster check: try all subsets of decreasing size."""
    n = len(ambient)
    # For n=12, we can be smart. Start from size 12 down.
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    s = ambient[combo[i]] + ambient[combo[j]]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    # Initial candidate
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    random.seed(42)
    
    # Strategy: random search + local improvement
    # A good ambient set should have many sum collisions
    # Try to construct sets where many pairs share sums
    
    def evaluate(s):
        return _max_sidon_size_fast(s)
    
    # Generate random candidates
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Random 12-set from [0, 128]
        candidate = random.sample(range(129), 12)
        candidate.sort()
        
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate
            if best_score <= 4:  # Can't do much better than ~4-5
                # Do local improvement
                break
        
        # Also try local perturbation of best_set
        if iteration % 100 == 0:
            perturbed = best_set[:]
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val not in perturbed:
                perturbed[idx] = new_val
                perturbed.sort()
                p_score = evaluate(perturbed)
                if p_score < best_score:
                    best_score = p_score
                    best_set = perturbed
                    if best_score <= 4:
                        break
    
    # Local improvement phase
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(129):
                    if j in best_set:
                        continue
                    trial = best_set[:]
                    trial[i] = j
                    trial.sort()
                    s = evaluate(trial)
                    if s < best_score:
                        best_score = s
                        best_set = trial
                        improved = True
                        break
    
    return best_set
# EVOLVE-BLOCK-END
