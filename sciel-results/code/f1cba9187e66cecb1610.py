"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math
from collections import defaultdict

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

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

def _can_add_fast(x, S, pf_cache):
    """
    Check if x can be added to set S using precomputed prime factors.
    For each pair (b,c) in S, check:
      1. x does not divide b*c  -> for all primes p: v_p(x) <= v_p(b)+v_p(c) must be FALSE
      2. b does not divide x*c  -> for all primes p: v_p(b) <= v_p(x)+v_p(c) must be FALSE
      3. c does not divide x*b  -> for all primes p: v_p(c) <= v_p(x)+v_p(b) must be FALSE
    """
    pf_x = pf_cache[x]
    n = len(S)
    for i in range(n):
        b = S[i]
        pf_b = pf_cache[b]
        for j in range(i + 1, n):
            c = S[j]
            pf_c = pf_cache[c]
            
            # Check x | b*c: need exists prime p where v_p(x) > v_p(b)+v_p(c)
            for p, ex in pf_x.items():
                if ex > pf_b.get(p, 0) + pf_c.get(p, 0):
                    # x divides b*c is FALSE, so this is OK for condition 1
                    pass
            # Actually we need: x does NOT divide b*c
            # x divides b*c iff for ALL primes p: v_p(x) <= v_p(b)+v_p(c)
            # So x does NOT divide b*c iff EXISTS prime p: v_p(x) > v_p(b)+v_p(c)
            # Wait no: x | bc means for all p, v_p(x) <= v_p(b)+v_p(c)
            # We need x ∤ bc, i.e., NOT (for all p, v_p(x) <= v_p(b)+v_p(c))
            # i.e., EXISTS p such that v_p(x) > v_p(b)+v_p(c)
            
            # Let me just do direct computation for correctness
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

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

def _precompute_conflict_masks():
    """
    For each pair (i,j) in UNIVERSE (i<j), precompute which elements k would conflict
    when added alongside both i and j.
    conflict_mask[i][j] = set of k such that adding k to {i,j} creates a violation.
    
    Actually, let's precompute: for each triple (a,b,c) distinct, whether a|bc, b|ac, c|ab.
    We'll build a conflict graph: edge between i and j if there exists some k such that
    {i,j,k} is invalid. But that's not quite right either.
    
    Better: precompute for each pair (b,c) the set of a's that would be invalid.
    invalid_a_for_bc = {a : a | b*c or b | a*c or c | a*b}
    """
    # conflict[a][b][c] = True if {a,b,c} violates the constraint
    # We'll store: for each unordered pair {b,c}, which a values conflict
    # conflict_with_pair[(b,c)] = set of a that cannot coexist with both b and c
    
    conflict_with_pair = {}
    for i in range(2, NMAX + 1):
        for j in range(i + 1, NMAX + 1):
            bad_ks = set()
            for k in range(2, NMAX + 1):
                if k == i or k == j:
                    continue
                # Check if {i, j, k} violates
                if (i * j) % k == 0 or (i * k) % j == 0 or (j * k) % i == 0:
                    bad_ks.add(k)
            conflict_with_pair[(i, j)] = bad_ks
    return conflict_with_pair

def _greedy(order, deadline, conflict_with_pair=None):
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
        
        # 1. Try to add candidates
        if candidates:
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
            
        # 2. Try swaps
        if not S:
            break
            
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

def _primes_up_to(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _try_add_composites_to_primes(prime_set, deadline, conflict_with_pair):
    """
    Start with all primes, try to add composites one by one.
    Use conflict_with_pair to quickly check if a composite can be added.
    """
    S = list(prime_set)
    S_set = set(S)
    composites = [x for x in range(2, NMAX + 1) if not _is_prime(x) and x not in S_set]
    
    # Sort composites by how "independent" they are - fewer shared prime factors with existing set
    def conflict_count(x):
        count = 0
        for s in S:
            if math.gcd(x, s) > 1:
                count += 1
        return count
    
    composites.sort(key=lambda x: (conflict_count(x), -x))
    
    for x in composites:
        if time.time() > deadline:
            break
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute conflict masks for fast checking
    conflict_with_pair = _precompute_conflict_masks()

    best_S = None
    best_size = 0

    # Phase 1: Start with all primes as baseline
    primes = _primes_up_to(NMAX)
    S_primes = primes[:]
    # Verify primes are valid
    valid_primes = []
    for p in primes:
        if _can_add(p, valid_primes):
            valid_primes.append(p)
    
    if valid_primes and len(valid_primes) > best_size:
        best_S = valid_primes
        best_size = len(valid_primes)

    # Try to extend prime set with composites
    if time.time() < deadline - 20:
        S_extended = _try_add_composites_to_primes(valid_primes, deadline - 20, conflict_with_pair)
        if S_extended and len(S_extended) > best_size:
            best_S = S_extended
            best_size = len(S_extended)

    # Phase 2: Diverse Greedy Initializations
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

    # Phase 3: Local Search Improvement
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

    # Final validation
    result = [int(x) for x in best_S]
    return result
