"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import comb

def _max_sidon_size(subset):
    """Compute the largest subset of 'subset' with all pair sums (i<=j) distinct."""
    n = len(subset)
    if n <= 1:
        return n
    elems = sorted(subset)
    best = [0]
    
    def backtrack(idx, current, current_sums):
        if idx >= n:
            best[0] = max(best[0], len(current))
            return
        if len(current) + (n - idx) <= best[0]:
            return
        # Try not taking idx
        backtrack(idx + 1, current, current_sums)
        # Try taking idx
        elem = elems[idx]
        new_sums = set(current_sums)
        valid = True
        for j in range(len(current)):
            val = current[j] + elem
            if val in new_sums:
                valid = False
                break
            new_sums.add(val)
        if valid:
            val = elem + elem
            if val in new_sums:
                valid = False
            else:
                new_sums.add(val)
        if valid:
            backtrack(idx + 1, current + [elem], new_sums)
    
    backtrack(0, [], set())
    return best[0]

def _eval_candidate(cand):
    """Return the max Sidon subset size of the candidate."""
    return _max_sidon_size(cand)

def _valid(cand):
    """Check candidate is 12 distinct integers in [0,128]."""
    if len(cand) != 12:
        return False
    if len(set(cand)) != 12:
        return False
    return all(0 <= x <= 128 for x in cand)

def _count_duplicate_sums(cand):
    """Count the number of duplicate unordered pair sums (including diagonals).
    More duplicates means harder to find large Sidon subsets."""
    n = len(cand)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = cand[i] + cand[j]
            sums[s] = sums.get(s, 0) + 1
    # Count excess: for each sum that appears k times, there are k-1 duplicates
    dup_count = sum(k - 1 for k in sums.values() if k > 1)
    return dup_count

def _greedy_swap_maximize_collisions(cand, rng, max_swaps=200):
    """Greedily swap elements to maximize duplicate pair sums."""
    current = list(cand)
    current_dups = _count_duplicate_sums(current)
    best = list(current)
    best_dups = current_dups
    
    for _ in range(max_swaps):
        # Pick two positions to swap
        i = rng.randint(0, 11)
        j = rng.randint(0, 11)
        if i == j:
            continue
        
        # Try swapping
        trial = list(current)
        trial[i], trial[j] = trial[j], trial[i]
        if not _valid(trial):
            continue
        
        trial_dups = _count_duplicate_sums(trial)
        if trial_dups > current_dups:
            current = trial
            current_dups = trial_dups
            if current_dups > best_dups:
                best = list(current)
                best_dups = current_dups
        elif trial_dups == current_dups and rng.random() < 0.1:
            # Occasional neutral move to escape local optima
            current = trial
            current_dups = trial_dups
    
    return best

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin within 60s limit
    
    best_cand = None
    best_score = 99
    seen = set()
    
    def consider(cand):
        nonlocal best_cand, best_score
        if not _valid(cand):
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = list(cand)
    
    rng = random.Random(42)
    
    # Strategy 1: Structured patterns (keep from original)
    for a in range(8, 25):
        for b in range(1, min(a, 10)):
            cand = []
            for i in range(6):
                x1 = a * i
                x2 = a * i + b
                if x1 <= 128 and x2 <= 128:
                    cand.extend([x1, x2])
            if len(cand) == 12 and len(set(cand)) == 12:
                consider(cand)
        if time.time() > deadline:
            break
    
    for s1 in range(4, 20):
        for s2 in range(4, 20):
            for o1 in range(0, 8):
                for o2 in range(0, 8):
                    a = [o1 + s1 * i for i in range(6)]
                    b = [o2 + s2 * i for i in range(6)]
                    cand = sorted(set(a + b))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        consider(cand)
        if time.time() > deadline:
            break
    
    for rows in (2, 3, 4, 6):
        for cols in (2, 3, 4, 6):
            if rows * cols != 12:
                continue
            for rs in range(5, 25):
                for cs in range(2, 12):
                    cand = sorted(set([rs * r + cs * c for r in range(rows) for c in range(cols)]))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        consider(cand)
        if time.time() > deadline:
            break
    
    if best_cand is None:
        best_cand = list(range(12))
        best_score = _eval_candidate(best_cand)
    
    # Strategy 2: Probabilistic collision maximization
    # Sample random 12-element subsets, then greedily maximize duplicate sums
    num_samples = 500
    for sample_idx in range(num_samples):
        if time.time() > deadline:
            break
        
        # Generate a random 12-element subset of [0, 128]
        random_cand = sorted(rng.sample(range(0, 129), 12))
        
        # Greedily swap to maximize collisions
        improved = _greedy_swap_maximize_collisions(random_cand, rng, max_swaps=100)
        consider(improved)
    
    # Strategy 3: Targeted local search around best candidate
    # Focus on swaps that increase collision count
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        cand = list(best_cand)
        current_dups = _count_duplicate_sums(cand)
        
        # Try single-element replacement or swap
        move_type = rng.random()
        if move_type < 0.5:
            # Swap two elements
            i = rng.randint(0, 11)
            j = rng.randint(0, 11)
            if i != j:
                trial = list(cand)
                trial[i], trial[j] = trial[j], trial[i]
                if _valid(trial):
                    trial_dups = _count_duplicate_sums(trial)
                    if trial_dups > current_dups:
                        consider(trial)
        else:
            # Replace one element with a random value
            idx = rng.randint(0, 11)
            trial = list(cand)
            trial[idx] = rng.randint(0, 128)
            if _valid(trial):
                trial_dups = _count_duplicate_sums(trial)
                if trial_dups > current_dups:
                    consider(trial)
    
    # Final: more aggressive local search with higher acceptance threshold
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        cand = list(best_cand)
        current_dups = _count_duplicate_sums(cand)
        
        # Random perturbation: change 1-2 elements
        num_changes = rng.randint(1, 2)
        for _ in range(num_changes):
            idx = rng.randint(0, 11)
            cand[idx] = rng.randint(0, 128)
        
        if not _valid(cand):
            continue
        
        key = tuple(sorted(cand))
        if key in seen:
            continue
        seen.add(key)
        
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = cand
    
    return best_cand
# EVOLVE-BLOCK-END
