import time
import random
import math

N = 5000
DEADLINE = 290.0  # Safety margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def _is_admissible(S, primes):
    """
    Check if S is admissible.
    """
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return False
    return True

def _get_smallest_violated_prime(S, primes):
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _greedy_remove(S, primes, rng):
    """
    Remove elements to make S admissible.
    """
    S = list(S)
    while True:
        p = _get_smallest_violated_prime(S, primes)
        if p is None:
            break
        freq = {}
        for x in S:
            r = x % p
            freq[r] = freq.get(r, 0) + 1
        
        min_freq = min(freq.values())
        candidates = [r for r, c in freq.items() if c == min_freq]
        r_to_remove = rng.choice(candidates)
        
        S = [x for x in S if x % p != r_to_remove]
        
    return S

def _greedy_add(S, candidates, primes, rng):
    """
    Add elements from candidates to S if they maintain admissibility.
    Candidates is a list of elements not in S.
    """
    S = list(S)
    candidates = list(candidates)
    rng.shuffle(candidates)
    
    # Precompute covered residues for efficiency
    # We only care about primes <= len(S) + max_adds
    # Let's just check on the fly or maintain a dict.
    # For speed, let's maintain a dict of sets for primes <= N.
    covered = {}
    for p in primes:
        if p > N: break
        covered[p] = set(x % p for x in S)
    
    added_any = True
    while added_any and candidates:
        added_any = False
        new_candidates = []
        for x in candidates:
            new_n = len(S) + 1
            valid = True
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                if r not in covered[p]:
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            if valid:
                S.append(x)
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_candidates.append(x)
        candidates = new_candidates
        
    return S

def _local_search_12(S, complement, primes, rng, max_iters=100):
    """
    Try 1-out-2-in swaps to increase size.
    Remove x from S, try to add y, z from complement.
    If S - {x} + {y, z} is admissible, accept.
    """
    S = set(S)
    comp = list(complement)
    rng.shuffle(comp)
    
    # Optimization: Precompute residues for all elements?
    # That's 5000 * num_primes. Might be too much memory/time if done repeatedly.
    # Let's just check on the fly.
    
    improved = False
    for _ in range(max_iters):
        if not S: break
        x = rng.choice(list(S))
        S_no_x = S - {x}
        
        # Find two elements in comp that can be added to S_no_x
        # This is an O(|comp|^2) check in worst case if done naively.
        # We can be smarter:
        # 1. Check which elements in comp can be added to S_no_x individually.
        # 2. If at least 2 exist, pick 2 and check if they can be added together.
        
        addable = []
        for y in comp:
            if y in S_no_x: continue
            # Check if y can be added to S_no_x
            # S_no_x is admissible? No, S was admissible, removing x keeps it admissible (subset).
            # Wait, removing an element from an admissible set keeps it admissible.
            # So S_no_x is admissible.
            # We need to check if S_no_x U {y} is admissible.
            # This is the same check as in _greedy_add.
            
            # Quick check:
            # For each p <= len(S_no_x)+1, does y cover a new residue that completes the set?
            n_no_x = len(S_no_x)
            valid = True
            # We don't have 'covered' sets for S_no_x here.
            # Recalculating is expensive.
            
            # Let's use a helper that takes a set and an element.
            # To make this fast, we should precompute residues for all x in [0, N] for all primes?
            # 5000 * 670 primes ~ 3.3M operations. Feasible once.
            
            pass # Skipping complex precomputation in this thought block, will implement in code if needed.
            # Actually, 1-out-2-in is expensive. Let's stick to randomized greedy add with multiple passes/shuffles first.
            
        break # Placeholder for logic
    
    return S

def construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Precompute residues for all numbers for all relevant primes?
    # This might speed up _greedy_add and local search.
    # Let's try to optimize _greedy_add first.
    
    while time.time() - start_time < DEADLINE - 5:
        rng = random.Random()
        
        # Strategy 1: Full set start
        if best_size == 0 or rng.random() < 0.5:
            S_init = list(range(N + 1))
        else:
            # Strategy 2: Random subset start
            # Density between 0.5 and 0.9
            density = rng.uniform(0.5, 0.9)
            target_size = int(N * density)
            S_init = rng.sample(range(N + 1), target_size)
            
        S_greedy = _greedy_remove(S_init, PRIMES, rng)
        
        S_set = set(S_greedy)
        removed = [x for x in range(N + 1) if x not in S_set]
        
        S_final = _greedy_add(S_greedy, removed, PRIMES, rng)
        
        # Optional: Second pass of greedy add with shuffled removed?
        # _greedy_add already shuffles.
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
    return best_S
