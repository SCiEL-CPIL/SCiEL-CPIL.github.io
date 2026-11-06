import time
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 290
    primes = _primes(N + 1)
    
    # Precompute sizes of residue classes
    # size[p][r] = number of x in [0, N] such that x % p == r
    sizes = {}
    for p in primes:
        if p > N + 1:
            break
        # For x in [0, N], count residues
        # N // p + 1 or N // p
        # Let's just compute directly or use formula
        # x = k*p + r. 0 <= k*p + r <= N.
        # Max k = (N - r) // p. Count = (N - r) // p + 1 if r <= N % p else ...
        # Actually simpler:
        # For r in 0..p-1:
        #   count = 0
        #   for x in range(r, N+1, p): count += 1
        # This is O(N/p) per r, total O(N).
        # But we can do it analytically.
        # Let N = q*p + rem.
        # For r <= rem: count = q + 1
        # For r > rem: count = q
        # Check: N=10, p=3. q=3, rem=1.
        # r=0: 0,3,6,9 (4). q+1=4. Correct.
        # r=1: 1,4,7,10 (4). q+1=4. Correct.
        # r=2: 2,5,8 (3). q=3. Correct.
        # So:
        q, rem = divmod(N, p)
        arr = np.empty(p, dtype=np.int32)
        if rem >= 0: # always true
            arr[:rem+1] = q + 1
            if rem + 1 < p:
                arr[rem+1:] = q
        sizes[p] = arr
        
    # Precompute elements for each residue class?
    # We need to check if a class is "full" (all elements removed).
    # And we need to add/remove elements.
    # Representing E as a boolean array is good.
    # To check if class (p, r) is full:
    #   sum(E[x] for x in [0, N] if x%p==r) == sizes[p][r]
    # This is O(N/p).
    # With ~669 primes, and checking all classes for all primes is O(N * pi(N)) ~ 3.3M ops per check.
    # Too slow if done naively in a loop.
    
    # Better: Maintain counts of removed elements per residue class.
    # removed_count[p][r]
    # Update when adding/removing x:
    #   for each p in primes:
    #       removed_count[p][x % p] += delta
    
    # Initialization
    best_S = None
    best_k = 0
    
    # We will try a few restarts with different initializations or heuristics
    # But first, let's implement the core loop efficiently.
    
    # State:
    # E: set of removed elements
    # E_arr: np.array of bools
    # removed_count: dict p -> np.array of counts
    
    # To speed up "find unsatisfied prime", we can keep a list of unsatisfied primes.
    # When we add/remove, we update counts.
    # A prime p is satisfied if any r has removed_count[p][r] == sizes[p][r].
    # We can maintain a boolean array satisfied_prime[p].
    
    # Let's use a greedy removal strategy with local search (add-backs).
    
    # Strategy:
    # 1. Start with S = [0, N] (E = empty).
    # 2. While S is not admissible:
    #    a. Find a prime p <= |S| that is violated (all residues present).
    #    b. Choose a residue r for p to remove.
    #       Heuristic: Remove the residue class with the smallest size? Or the one that causes least "future" harm?
    #       Smallest size is usually good for keeping |S| large.
    #    c. Remove all x in S such that x % p == r.
    #    d. Update state.
    # 3. Local Search:
    #    Try to add back elements one by one if it doesn't break admissibility.
    #    Or, try to swap: remove x, add y.
    
    # Let's implement a robust version of this.
    
    def try_greedy_removal(seed_offset=0):
        # Use seed_offset to vary tie-breaking if needed, but deterministic is fine first.
        E_arr = np.zeros(N + 1, dtype=np.bool_)
        removed_count = {}
        satisfied = {}
        
        for p in primes:
            if p > N + 1:
                break
            removed_count[p] = np.zeros(p, dtype=np.int32)
            satisfied[p] = False
            
        # Initial state: S = all, E = empty.
        # No primes are satisfied (removed_count is 0, sizes > 0).
        
        k = N + 1
        
        # Loop until admissible
        # We need to find violated primes.
        # A prime p is violated if NOT satisfied[p] AND p <= k.
        
        # To find violated primes quickly, we can iterate through primes.
        # Number of primes is ~669.
        
        while time.time() < deadline - 5:
            # Find violated primes
            violated = []
            for p in primes:
                if p > k:
                    break
                if not satisfied[p]:
                    violated.append(p)
            
            if not violated:
                break
            
            # Pick a violated prime
            # Heuristic: Smallest prime is most restrictive?
            # Or pick the one with the "largest" residue class to remove?
            # Removing a large class reduces k significantly.
            # We want to minimize reduction in k.
            # So we should pick a prime p, and a residue r such that the number of elements removed is small.
            # The number of elements in residue r mod p that are currently in S is:
            #   current_count[p][r] = sizes[p][r] - removed_count[p][r]
            # We want to minimize current_count[p][r] over all violated p and all r.
            
            min_count = float('inf')
            best_p = -1
            best_r = -1
            
            for p in violated:
                # For each residue r, compute elements currently in S
                # current_counts = sizes[p] - removed_count[p]
                # We want min(current_counts)
                # Note: if current_counts[r] == 0, it's already removed (satisfied), but p is violated so this shouldn't happen?
                # Wait, if p is violated, it means NO residue is fully removed.
                # So current_counts[r] > 0 for all r.
                
                curr = sizes[p] - removed_count[p]
                # Find min curr
                min_curr = np.min(curr)
                if min_curr < min_count:
                    min_count = min_curr
                    # Find index of min
                    # If multiple, pick the first? Or random?
                    best_r = int(np.argmin(curr))
                    best_p = p
            
            # Remove residue best_r mod best_p
            p_to_remove = best_p
            r_to_remove = best_r
            
            # Iterate through elements in this residue class
            # x = r, r+p, r+2p, ...
            for x in range(r_to_remove, N + 1, p_to_remove):
                if not E_arr[x]:
                    E_arr[x] = True
                    # Update removed_count for all primes q
                    for q in primes:
                        if q > N + 1:
                            break
                        idx = x % q
                        removed_count[q][idx] += 1
                        # Check if q is now satisfied
                        if not satisfied[q]:
                            if removed_count[q][idx] == sizes[q][idx]:
                                satisfied[q] = True
                                # Optimization: if q is satisfied, we don't need to track it as violated anymore
                                # But we still need to update counts for future add-backs?
                                # Yes, but we can skip the "satisfied" check for already satisfied primes.
            
            k = N + 1 - int(np.sum(E_arr))
            
            # If k dropped a lot, some primes might become inactive (p > k).
            # This is handled in the next loop's "violated" check.
        
        # Now S is admissible.
        # Try to improve by adding back elements.
        # Local Search:
        # Iterate through elements in E (removed ones).
        # Try to add x back.
        # Adding x back means:
        #   E_arr[x] = False
        #   removed_count[q][x%q] -= 1 for all q
        #   Check if any prime q becomes violated (i.e., was satisfied, but now the class x%q is no longer full).
        #   If so, we cannot add x back (unless we remove something else, which is more complex).
        #   If no prime becomes violated, we keep x in S.
        
        # To make this efficient, we can check which primes are "critical" for x.
        # A prime q is critical for x if removed_count[q][x%q] == sizes[q][x%q] (i.e., the class is full).
        # If we remove x from E, that count drops by 1, so the class is no longer full.
        # If q <= k, then q becomes violated.
        # So we can only add x back if for all q <= k, the class x%q is NOT full.
        # i.e., removed_count[q][x%q] < sizes[q][x%q] for all q <= k.
        
        # Let's iterate through candidates for add-back.
        # We can iterate through all x in [0, N] that are in E.
        # There are ~1000-2000 such elements.
        # For each, check all q <= k.
        # k is ~3000-4000. Primes ~ 500.
        # 2000 * 500 = 1,000,000 checks.
        # Each check is O(1) array access.
        # This is fast enough.
        
        # We might do multiple passes of add-back.
        
        # Actually, we can just iterate through all x in [0, N] and try to add them if they are in E.
        # Order: maybe by x? Or random?
        
        # Let's do a single pass first.
        
        for x in range(N + 1):
            if E_arr[x]:
                # Try to add back
                can_add = True
                for q in primes:
                    if q > k:
                        break
                    if removed_count[q][x % q] == sizes[q][x % q]:
                        can_add = False
                        break
                
                if can_add:
                    E_arr[x] = False
                    # Update counts
                    for q in primes:
                        if q > N + 1:
                            break
                        idx = x % q
                        removed_count[q][idx] -= 1
                        # Note: q was satisfied, so removing x makes it unsatisfied?
                        # No, we only add back if it WASN'T the critical element?
                        # Wait.
                        # If removed_count[q][idx] == sizes[q][idx], then the class is full.
                        # If we remove x from E, count becomes sizes-1.
                        # So the class is NO LONGER full.
                        # So q becomes VIOLATED (if q <= k).
                        # My condition `if removed_count[q][x % q] == sizes[q][x % q]: can_add = False`
                        # is correct. If it IS full, we CANNOT add x back, because it would break admissibility.
                        # If it is NOT full, we CAN add x back, and it remains admissible (since it wasn't full before, and removing x doesn't make it full).
                        # Wait, does removing x from E make the class "less full"? Yes.
                        # So if it was not full, it remains not full.
                        # So the prime remains violated?
                        # No, the prime was violated (not satisfied).
                        # If we add x back, the class becomes even less full.
                        # So the prime remains violated.
                        # This is a problem!
                        # If a prime is violated, we MUST satisfy it.
                        # So we can only add x back if it does not cause any prime to become violated.
                        # But if a prime is already violated, adding x back doesn't fix it, nor does it break it (it stays violated).
                        # But we need the set to be ADMISSIBLE.
                        # Admissible means ALL primes are satisfied.
                        # So if any prime is violated, the set is INVALID.
                        # Therefore, we can ONLY add x back if the set is CURRENTLY admissible (all primes satisfied).
                        # And adding x back must NOT violate any prime.
                        # A prime q is violated after adding x back if:
                        #   1. q <= k_new
                        #   2. The class x%q was FULL (so removing x makes it not full)
                        #   3. No other class in q is full.
                        # Wait, if q was satisfied, it means AT LEAST ONE class was full.
                        # If we remove x from a full class, that class becomes not full.
                        # If q has ANOTHER full class, q remains satisfied.
                        # If q has ONLY that one full class, q becomes violated.
                        
                        # So the condition for adding x back is:
                        # For all q <= k, if removed_count[q][x%q] == sizes[q][x%q],
                        # then there must be ANOTHER residue r' != x%q such that removed_count[q][r'] == sizes[q][r'].
                        
                        # This is more complex.
                        # Let's re-evaluate.
                        
                        # Current state: Admissible. All q <= k are satisfied.
                        # We want to add x back.
                        # For each q <= k:
                        #   If x%q was a full class (removed_count == size):
                        #     After removal, count decreases.
                        #     We need to check if q is still satisfied by another class.
                        #     If not, q becomes violated -> cannot add x.
                        #   If x%q was not a full class:
                        #     After removal, count decreases further.
                        #     q was already satisfied by some other class (or this one? No, this one wasn't full).
                        #     So q remains satisfied.
                        
                        # So we need to check for each q where x%q is full, if there's another full class.
                        
                        # To do this efficiently, we can precompute for each q the number of full classes.
                        # full_count[q] = number of r such that removed_count[q][r] == sizes[q][r].
                        # If full_count[q] >= 2, then removing x from any full class is safe.
                        # If full_count[q] == 1, then removing x from the unique full class is UNSAFE.
                        # If full_count[q] == 0, q is not satisfied (contradiction, since we are in admissible state).
                        
                        # So we maintain full_count[q].
                        # Update full_count when counts change.
                        
                    # Let's refine the loop.
                    
                # Reset can_add logic
                pass # The pass above was a placeholder. I will rewrite the inner loop properly.
            
            # End if E_arr[x]
        
        # The logic above was getting messy. Let's restart the local search part cleanly.
        
        # State: E_arr, removed_count, sizes, satisfied (implied by full_count > 0)
        # We need full_count.
        
        # Let's re-initialize and do the whole thing again with full_count.
        
        pass # This function is a mess. I will rewrite construct() entirely in the final code block.
        
        return [x for x in range(N + 1) if not E_arr[x]]

    # Since the function above is incomplete/messy, I will discard it and write the final version below.
    
    # Final Version Logic:
    # 1. Greedy Removal to get a valid S.
    # 2. Local Search (Add-back) to maximize |S|.
    
    # Implementation details:
    # - Use numpy arrays for counts.
    # - Maintain full_count[p] for each prime.
    # - Greedy: Find violated p. Pick r with min current_count. Remove. Update counts and full_count.
    # - Local Search: Iterate x in E. Check if adding x back is safe.
    #   Safe if for all q <= k:
    #     If x%q is a full class (count==size), then full_count[q] >= 2.
    #   If safe, add x back. Update counts and full_count.
    
    # Let's implement this.
    
    def run_search():
        E_arr = np.zeros(N + 1, dtype=np.bool_)
        removed_count = {}
        full_count = {}
        
        for p in primes:
            if p > N + 1:
                break
            removed_count[p] = np.zeros(p, dtype=np.int32)
            full_count[p] = 0
            
        k = N + 1
        
        # Phase 1: Greedy Removal
        while time.time() < deadline - 20:
            # Find violated primes (full_count[p] == 0 and p <= k)
            violated = []
            for p in primes:
                if p > k:
                    break
                if full_count[p] == 0:
                    violated.append(p)
            
            if not violated:
                break
            
            # Pick best p, r
            min_count = float('inf')
            best_p = -1
            best_r = -1
            
            for p in violated:
                curr = sizes[p] - removed_count[p]
                # curr[r] is the number of elements in S for residue r
                # We want to remove the residue with the FEWEST elements in S?
                # Yes, to minimize loss of |S|.
                min_curr = np.min(curr)
                if min_curr < min_count:
                    min_count = min_curr
                    best_r = int(np.argmin(curr))
                    best_p = p
            
            p_to_remove = best_p
            r_to_remove = best_r
            
            for x in range(r_to_remove, N + 1, p_to_remove):
                if not E_arr[x]:
                    E_arr[x] = True
                    for q in primes:
                        if q > N + 1:
                            break
                        idx = x % q
                        old_count = removed_count[q][idx]
                        removed_count[q][idx] += 1
                        new_count = removed_count[q][idx]
                        
                        if old_count == sizes[q][idx]:
                            # Was full, now not full? No, old_count == size means it WAS full.
                            # New count is size + 1? Impossible, max count is size.
                            # Wait, removed_count cannot exceed size.
                            # If old_count == size, it was full.
                            # If we add x, and x was NOT in E, then x is in the class.
                            # So the class was already full?
                            # If the class was full, it means ALL elements in the class were in E.
                            # So x was in E.
                            # But we check `if not E_arr[x]`.
                            # So x was NOT in E.
                            # Contradiction?
                            # No. "Full" means removed_count == size.
                            # This means all elements of the class are in E.
                            # So x MUST be in E.
                            # So `if not E_arr[x]` would be False.
                            # So we never update a full class to be "more full".
                            # So old_count == size implies we don't enter this block.
                            pass
                        elif new_count == sizes[q][idx]:
                            # Just became full
                            full_count[q] += 1
                        # If old_count < size and new_count < size, full_count unchanged.
                        # If old_count < size and new_count == size, full_count increases.
            
            k = N + 1 - int(np.sum(E_arr))
        
        # Phase 2: Local Search (Add-back)
        # Iterate through all x in [0, N].
        # We can do multiple passes.
        
        for _ in range(3): # 3 passes
            # Collect candidates
            candidates = [x for x in range(N + 1) if E_arr[x]]
            # Shuffle? Or sort?
            # Let's try sorting by x.
            for x in candidates:
                if not E_arr[x]:
                    continue # Already added back
                
                can_add = True
                for q in primes:
                    if q > k:
                        break
                    idx = x % q
                    if removed_count[q][idx] == sizes[q][idx]:
                        # This class is full.
                        # If we add x back, it becomes not full.
                        # We need full_count[q] >= 2 to ensure q remains satisfied.
                        if full_count[q] < 2:
                            can_add = False
                            break
                
                if can_add:
                    E_arr[x] = False
                    for q in primes:
                        if q > N + 1:
                            break
                        idx = x % q
                        old_count = removed_count[q][idx]
                        removed_count[q][idx] -= 1
                        new_count = removed_count[q][idx]
                        
                        if old_count == sizes[q][idx]:
                            # Was full, now not full
                            full_count[q] -= 1
                        # If old_count > size, impossible.
                        # If old_count < size, full_count unchanged.
                    
                    k += 1 # k increases by 1
        
        S = [x for x in range(N + 1) if not E_arr[x]]
        return S

    S = run_search()
    
    # Verify
    k = len(S)
    S_set = set(S)
    valid = True
    for p in primes:
        if p > k:
            break
        residues_seen = set()
        for x in S_set:
            residues_seen.add(x % p)
        if len(residues_seen) == p:
            valid = False
            break
            
    if valid:
        return S
    else:
        # Fallback
        return list(range(N + 1))

# EVOLVE-BLOCK-END
