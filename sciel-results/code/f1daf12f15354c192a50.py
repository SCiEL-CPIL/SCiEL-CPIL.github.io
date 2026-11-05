"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

def _pair_sums_with_diag(subset):
    """Return list of all unordered pair sums including diagonals (i<=j)."""
    sums = []
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            sums.append(subset[i] + subset[j])
    return sums

def _count_collisions(sums):
    """Count number of duplicate sums (excess over unique count)."""
    c = Counter(sums)
    return sum(v - 1 for v in c.values() if v > 1)

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
        # Try not taking idx first (to find smaller subsets quickly for pruning)
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
    
    # Strategy: Use probabilistic collision-maximization
    # Sample random 12-element subsets and greedily swap to maximize duplicate pair sums
    
    rng = random.Random(12345)
    
    # First, try some structured candidates as initial seeds
    structured_seeds = []
    
    # 6 pairs with small stride
    for a in range(6, 20):
        for b in range(1, min(a, 8)):
            cand = []
            for i in range(6):
                x1 = a * i
                x2 = a * i + b
                if x1 <= 128 and x2 <= 128:
                    cand.extend([x1, x2])
            if len(cand) == 12 and len(set(cand)) == 12:
                structured_seeds.append(cand)
    
    # Grid structures
    for rows in (2, 3, 4, 6):
        for cols in (2, 3, 4, 6):
            if rows * cols != 12:
                continue
            for rs in range(5, 25):
                for cs in range(2, 12):
                    cand = sorted(set([rs * r + cs * c for r in range(rows) for c in range(cols)]))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        structured_seeds.append(cand)
    
    # Two-AP unions
    for s1 in range(4, 15):
        for s2 in range(4, 15):
            for o1 in range(0, 6):
                for o2 in range(0, 6):
                    a = [o1 + s1 * i for i in range(6)]
                    b = [o2 + s2 * i for i in range(6)]
                    cand = sorted(set(a + b))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        structured_seeds.append(cand)
    
    # Evaluate structured seeds
    for cand in structured_seeds:
        consider(cand)
        if time.time() > deadline:
            break
    
    if best_cand is None:
        best_cand = list(range(12))
        best_score = _eval_candidate(best_cand)
    
    # Now do probabilistic collision-maximization local search
    # Objective: maximize number of duplicate pair sums (proxy for minimizing max Sidon size)
    
    def collision_score(cand):
        sums = _pair_sums_with_diag(cand)
        return _count_collisions(sums)
    
    best_cand_coll = collision_score(best_cand)
    
    # Multiple random restarts with greedy swap
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best_cand or random
        if restart == 0:
            cand = list(best_cand)
        else:
            cand = sorted(rng.sample(range(0, 129), 12))
        
        coll = collision_score(cand)
        
        # Greedy swap: try swapping each element with a random new value
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order for diversity
            indices = list(range(12))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                old_val = cand[idx]
                # Try several random replacements
                for _ in range(20):
                    new_val = rng.randint(0, 128)
                    if new_val == old_val:
                        continue
                    if new_val in cand:
                        continue
                    cand[idx] = new_val
                    new_coll = collision_score(cand)
                    if new_coll > coll:
                        coll = new_coll
                        improved = True
                        break
                    else:
                        cand[idx] = old_val
                # Also try swapping with another element's value
                for jdx in range(12):
                    if jdx == idx:
                        continue
                    if time.time() > deadline:
                        break
                    cand[idx], cand[jdx] = cand[jdx], cand[idx]
                    new_coll = collision_score(cand)
                    if new_coll > coll:
                        coll = new_coll
                        improved = True
                        break
                    else:
                        cand[idx], cand[jdx] = cand[jdx], cand[idx]
        
        # Evaluate the candidate's actual max Sidon size
        if _valid(cand):
            s = _eval_candidate(cand)
            if s < best_score:
                best_score = s
                best_cand = list(cand)
                best_cand_coll = collision_score(best_cand)
    
    # Final polish: more targeted local search around best candidate
    if time.time() < deadline:
        cand = list(best_cand)
        for iteration in range(2000):
            if time.time() > deadline:
                break
            cand_copy = list(cand)
            # Perturb 1-2 elements
            for _ in range(rng.randint(1, 2)):
                idx = rng.randint(0, 11)
                cand_copy[idx] = rng.randint(0, 128)
            if not _valid(cand_copy):
                continue
            key = tuple(sorted(cand_copy))
            if key in seen:
                continue
            seen.add(key)
            s = _eval_candidate(cand_copy)
            if s < best_score:
                best_score = s
                best_cand = cand_copy
    
    return best_cand
# EVOLVE-BLOCK-END
