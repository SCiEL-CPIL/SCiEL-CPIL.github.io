"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of ints.
    """
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _greedy_conflict_priority(deadline):
    """
    Greedy that at each step picks the candidate with minimum 'conflict potential'.
    Conflict potential of x relative to current set S: number of pairs (b,c) in S
    where x would create a conflict. We approximate by counting how many elements
    in S share a factor with x (as a proxy).
    """
    S = []
    S_set = set()
    candidates = list(UNIVERSE)
    
    while candidates and time.time() < deadline:
        # Score each candidate: lower is better (fewer conflicts)
        best_x = None
        best_score = float('inf')
        
        for x in candidates:
            # Count elements in S that share a factor with x
            shared = 0
            for s in S:
                if math.gcd(x, s) > 1:
                    shared += 1
            # Also consider: how many elements in S could form a product divisible by x
            # or where x could divide their product with another element
            # Use shared as a proxy
            score = shared
            if score < best_score:
                best_score = score
                best_x = x
            elif score == best_score:
                # Tie-break: prefer larger x
                if x > best_x:
                    best_x = x
        
        if best_x is None:
            break
        
        if _can_add(best_x, S):
            S.append(best_x)
            S_set.add(best_x)
            candidates.remove(best_x)
        else:
            candidates.remove(best_x)
    
    return S

def _local_search_improve(S, deadline, max_iters=500, rng_seed=42):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        progress = False
        
        # 1. Try to add candidates - try more candidates per iteration
        if candidates:
            # Try up to 30 random candidates
            n_try = min(30, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            for x in trial_candidates:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break
            
        if progress:
            continue
            
        # 2. Try swaps - try multiple removals if first doesn't work
        if not S:
            break
            
        # Try up to 5 different removals
        for _attempt in range(5):
            if time.time() > deadline:
                break
                
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            if not candidates:
                continue
                
            # Try up to 30 candidates for replacement
            n_try = min(30, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            swapped = False
            for x in trial_candidates:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            
            if swapped:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_S = None
    best_size = 0

    # Phase 1: Diverse Greedy Initializations
    # 1. Largest First
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy(order_largest, deadline)
    if S1 and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    # 2. Smallest First
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy(order_smallest, deadline)
    if S2 and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)

    # 3. Conflict-priority greedy
    if time.time() < deadline - 15:
        S3 = _greedy_conflict_priority(deadline - 15)
        if S3 and len(S3) > best_size:
            best_S = S3
            best_size = len(S3)

    # 4. Randomized Orders
    for seed in [1, 7, 42, 100, 200, 500, 777]:
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy(order_rand, deadline)
        if Sr and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)

    if not best_S:
        best_S = [2, 3]
        best_size = 2

    # Phase 2: Local Search Improvement
    # Run 1
    if time.time() < deadline - 12:
        S_improved = _local_search_improve(best_S, deadline - 12, max_iters=3000, rng_seed=42)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    # Run 2 on perturbed
    if time.time() < deadline - 6:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
        S_improved2 = _local_search_improve(S_perturbed, deadline - 6, max_iters=1500, rng_seed=99)
        if S_improved2 and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Run 3 on another perturbed
    if time.time() < deadline - 2:
        rng_perturb2 = random.Random(456)
        S_perturbed2 = best_S[:]
        for _ in range(rng_perturb2.randint(1, 2)):
            if S_perturbed2:
                idx = rng_perturb2.randrange(len(S_perturbed2))
                S_perturbed2.pop(idx)
        S_improved3 = _local_search_improve(S_perturbed2, deadline - 2, max_iters=800, rng_seed=200)
        if S_improved3 and len(S_improved3) > best_size:
            best_S = S_improved3
            best_size = len(S_improved3)

    return [int(x) for x in best_S]
