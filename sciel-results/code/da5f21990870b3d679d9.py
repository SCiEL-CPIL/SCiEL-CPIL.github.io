import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Initial set: 1 and primes > sqrt(N)
    sqrt_n = int(N ** 0.5)
    primes = _primes(N)
    S = set([1] + [p for p in primes if p > sqrt_n])
    
    # Precompute primes up to N (for checking)
    all_primes = _primes(N)
    
    # Data structures for admissibility
    # We need to handle primes up to len(S).
    # len(S) will grow, so we need primes up to N.
    
    # Initialize counts
    # counts[p][r] for p in all_primes
    # This is memory heavy? 669 primes. Max r is p-1.
    # Total entries sum(p) ~ 1.4M. OK.
    
    # But we only care about p <= len(S).
    # Let's maintain a list of active primes.
    
    # Better: Just use a dictionary or list of lists.
    # Since N is small, we can just compute on the fly?
    # No, 5000 candidates * 118 primes is fine.
    
    # Let's use the incremental update method.
    
    # Map prime to index
    prime_index = {p: i for i, p in enumerate(all_primes)}
    
    # counts[i][r]
    # We can't allocate 669 * 5000 easily?
    # 669 * 5000 * 4 bytes = 13 MB. OK.
    # But r < p. So variable size.
    
    # Let's just use a list of dicts or arrays.
    # For speed, use list of lists, initialized to 0.
    # max_p = N
    # counts = [[0]*p for p in all_primes] -> List of lists of different lengths.
    
    # Actually, we only need to check primes p <= len(S).
    # len(S) starts at ~650.
    
    # Let's simplify:
    # For each candidate x, check if adding it breaks admissibility.
    # Check: For all p <= len(S)+1, is there a missing class?
    
    # To make it fast, we maintain `missing` set of primes.
    # Initially, for the seed set, we compute which primes are "full" (none missing).
    # If any prime p <= len(S) is full, the set is NOT admissible.
    # But the seed is admissible.
    
    # Let's compute the state for the seed.
    def get_state(S_set):
        size = len(S_set)
        # Primes to check
        check_primes = [p for p in all_primes if p <= size]
        
        # For each p, count residues
        # Return True if admissible
        for p in check_primes:
            residues = set()
            for x in S_set:
                residues.add(x % p)
            if len(residues) == p:
                return False # Full
        return True

    # This is O(|S| * pi(|S|)).
    # 650 * 118 = 76k.
    # 4350 candidates * 76k = 330M. Too slow.
    
    # Use incremental update.
    
    # State:
    # counts[p][r]
    # missing_count[p]
    
    # Initialize for S
    size = len(S)
    check_primes = [p for p in all_primes if p <= size]
    
    # We need to extend check_primes as size grows.
    
    # Let's just use the incremental logic carefully.
    
    # Data:
    # counts: dict of dict? Or list of arrays.
    # Let's use list of arrays for primes up to N.
    # counts[i] is an array of size p_i.
    
    # But we only update for p <= current_size.
    
    # Initialize
    counts = [None] * len(all_primes)
    missing_count = [0] * len(all_primes)
    
    for i, p in enumerate(all_primes):
        counts[i] = [0] * p
        missing_count[i] = p # Initially all missing
        
    # Update with seed S
    for x in S:
        for i, p in enumerate(all_primes):
            if p > len(S): # Optimization: only update if p is relevant?
                # But p might become relevant later.
                # So we must update all p?
                # No, if p > len(S), we don't care about its state yet.
                # But when size grows, we will.
                # So we MUST maintain counts for all p <= N.
                # 669 primes * 5000 elements = 3.3M updates.
                # 3.3M * (mod + access) is fast.
                pass
            
            r = x % p
            if counts[i][r] == 0:
                missing_count[i] -= 1
            counts[i][r] += 1
            
    # Now S is admissible if missing_count[i] > 0 for all i where p_i <= len(S).
    
    # Candidate pool
    candidates = [x for x in range(N + 1) if x not in S]
    
    # Sort candidates?
    # Maybe sort by "cost"?
    # Cost of adding x: number of primes p where x fills the last missing class.
    # If cost > 0, it's a violation.
    # We want cost == 0.
    
    # Greedy: Add x if it doesn't violate.
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Check if adding x violates
        size = len(S)
        # New size will be size + 1
        # Primes to check: p <= size + 1
        
        violates = False
        # Store changes to revert if violation
        changes = []
        
        for i, p in enumerate(all_primes):
            if p > size + 1:
                break # Primes larger than new size don't matter
            
            r = x % p
            if counts[i][r] == 0:
                # This was the last missing class
                missing_count[i] -= 1
                changes.append((i, r))
                if missing_count[i] == 0:
                    violates = True
                    break
            else:
                # Not the last missing, just increase count
                changes.append((i, r))
            
            counts[i][r] += 1
        
        if not violates:
            S.add(x)
            # Accept changes
        else:
            # Revert
            for i, r in changes:
                counts[i][r] -= 1
                if counts[i][r] == 0:
                    missing_count[i] += 1
            # Note: if we broke early, we only updated some.
            # We need to revert exactly what we did.
            # My loop above breaks, so `changes` only has the ones processed.
            # But `counts` was incremented for them.
            # So revert is correct.
            
    return list(S)
