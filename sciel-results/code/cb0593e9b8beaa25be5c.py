"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _prime_factors(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute prime factors for all numbers in universe
_FACTORS = {}
for _x in UNIVERSE:
    _FACTORS[_x] = _prime_factors(_x)

def _can_add(x, S):
    """
    Check if x can be added to set S.
    For every pair (b, c) in S with b != c, check that x does not divide b*c.
    Also check that for every pair (b, c) in S, b does not divide x*c and c does not divide x*b.
    """
    n = len(S)
    fx = _FACTORS[x]
    
    for i in range(n):
        b = S[i]
        fb = _FACTORS[b]
        for j in range(i + 1, n):
            c = S[j]
            fc = _FACTORS[c]
            
            # Check x | b*c: for each prime p in fx, need fx[p] <= fb.get(p,0) + fc.get(p,0)
            conflict = False
            for p, exp_x in fx.items():
                if exp_x > fb.get(p, 0) + fc.get(p, 0):
                    conflict = True
                    break
            if conflict:
                return False
            
            # Check b | x*c: for each prime p in fb, need fb[p] <= fx.get(p,0) + fc.get(p,0)
            for p, exp_b in fb.items():
                if exp_b > fx.get(p, 0) + fc.get(p, 0):
                    return False
            
            # Check c | x*b: for each prime p in fc, need fc[p] <= fx.get(p,0) + fb.get(p,0)
            for p, exp_c in fc.items():
                if exp_c > fx.get(p, 0) + fb.get(p, 0):
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
    Greedy that at each step picks the candidate with minimum conflict potential.
    """
    S = []
    S_set = set()
    candidates = list(UNIVERSE)
    
    while candidates and time.time() < deadline:
        best_x = None
        best_score = float('inf')
        
        for x in candidates:
            shared = 0
            for s in S:
                if math.gcd(x, s) > 1:
                    shared += 1
            score = shared
            if score < best_score:
                best_score = score
                best_x = x
            elif score == best_score:
                if x > best_x:
                    best_x = x
        
        if best_x is None:
            break
        
        if _can_add(best_x, S):
            S.append(best_x)
            S_set.add(best_x)
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
        
        # 1. Try to add candidates
        if candidates:
            n_try = min(40, len(candidates))
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
            
        # 2. Try swaps
        if not S:
            break
            
        for _attempt in range(6):
            if time.time() > deadline:
                break
                
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            if not candidates:
                continue
                
            n_try = min(40, len(candidates))
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
    deadline = start_time + 85.0

    best_S = None
    best_size = 0

    # Phase 1: Diverse Greedy Initializations
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy(order_largest, deadline)
    if S1 and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy(order_smallest, deadline)
    if S2 and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)

    if time.time() < deadline - 10:
        S3 = _greedy_conflict_priority(deadline - 10)
        if S3 and len(S3) > best_size:
            best_S = S3
            best_size = len(S3)

    for seed in [1, 7, 42, 100, 200, 500, 777, 1234, 9999]:
        if time.time() > deadline - 15:
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
    if time.time() < deadline - 10:
        S_improved = _local_search_improve(best_S, deadline - 10, max_iters=4000, rng_seed=42)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    if time.time() < deadline - 5:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
        S_improved2 = _local_search_improve(S_perturbed, deadline - 5, max_iters=2000, rng_seed=99)
        if S_improved2 and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    if time.time() < deadline - 2:
        rng_perturb2 = random.Random(456)
        S_perturbed2 = best_S[:]
        for _ in range(rng_perturb2.randint(1, 2)):
            if S_perturbed2:
                idx = rng_perturb2.randrange(len(S_perturbed2))
                S_perturbed2.pop(idx)
        S_improved3 = _local_search_improve(S_perturbed2, deadline - 2, max_iters=1000, rng_seed=200)
        if S_improved3 and len(S_improved3) > best_size:
            best_S = S_improved3
            best_size = len(S_improved3)

    # Final validation
    result = [int(x) for x in best_S]
    assert len(result) > 0
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result)
    assert all(2 <= x <= 256 for x in result)
    assert len(set(result)) == len(result)
    
    return result
