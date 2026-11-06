"""Combined approach: modular sieve pre-filtering + bitset DP validation + descending greedy."""
import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality.
    dp[k] is an integer where bit s is set if sum s is achievable by k elements.
    """
    n = len(elements)
    if n == 0:
        return [0]
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    return dp

def _modular_sieve_check(x, selected, primes):
    """Quick pre-filter: for each prime p, check if 2x mod p equals (a+b) mod p
    for any pair a,b in selected. If so, x might be midpoint of a,b (necessary
    but not sufficient condition for violation). Returns False if any prime
    shows a potential conflict."""
    n = len(selected)
    if n < 2:
        return True  # Can't form pairs
    
    for p in primes:
        # Build set of all pair sums mod p from selected
        pair_sums = set()
        for i in range(n):
            ai = selected[i] % p
            for j in range(i + 1, n):
                bj = selected[j] % p
                pair_sums.add((ai + bj) % p)
        
        # Check if 2x mod p is in pair_sums
        if (2 * x) % p in pair_sums:
            return False  # Potential conflict detected
    
    return True

def _is_valid_mean_free_with_dp(A, x):
    """Exact check using bitset DP. Returns True if adding x keeps set mean-free."""
    n = len(A)
    
    # 1. Check if x is the mean of any subset of A of size k >= 2
    if n >= 2:
        dp_A = _build_dp(A)
        for k in range(2, n + 1):
            target = k * x
            if (dp_A[k] >> target) & 1:
                return False
            
    # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a}
    # Subset must contain x. Sum(T) + x = k*a, T subset of A\{a}, |T|=k-1=j
    # Sum(T) = k*a - x, j = k-1
    for a in A:
        others = [v for v in A if v != a]
        if len(others) < 1:
            continue
        
        dp_others = _build_dp(others)
        max_k = len(others) + 1  # subset of (A U {x})\{a} has size up to n
        for k in range(2, max_k + 1):
            j = k - 1
            if j > len(others):
                break
            target_sum = k * a - x
            if target_sum < 0:
                continue
            if (dp_others[j] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Small primes for modular sieve
    primes = [3, 5, 7, 11, 13]
    
    best_set = []
    best_len = 0
    
    def try_order(order):
        nonlocal best_set, best_len
        current_set = []
        for x in order:
            if time.time() > deadline:
                break
            # Quick modular sieve pre-filter
            if not _modular_sieve_check(x, current_set, primes):
                continue
            # Exact DP validation
            if _is_valid_mean_free_with_dp(current_set, x):
                current_set.append(x)
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                if best_len >= 48:
                    return True
        return best_len >= 48
    
    # Strategy 1: Descending greedy (primary - larger numbers have fewer constraints)
    candidates_desc = list(range(512, 0, -1))
    if try_order(candidates_desc):
        return best_set
    
    # Strategy 2: Modular-ordered greedy
    # Sort candidates by their residue patterns to find compatible sets
    def modular_key(x):
        # Prioritize numbers with "unusual" residues
        return tuple(x % p for p in primes)
    
    candidates_mod = sorted(range(1, 513), key=modular_key, reverse=True)
    if try_order(candidates_mod):
        return best_set
    
    # Strategy 3: Random restarts with descending bias
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Create biased order: mix of descending and random
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        # Sort with bias toward larger values
        candidates.sort(key=lambda x: -x + rng.random() * 50)
        
        if try_order(candidates):
            return best_set
    
    # Strategy 4: Pure random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        if try_order(candidates):
            return best_set
    
    return best_set
