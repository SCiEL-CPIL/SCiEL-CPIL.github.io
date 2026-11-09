import time
import numpy as np
import math

N = 5000
MAX_TIME = 295.0  # Safety margin

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

# Primes up to N (since max tuple size is N+1, primes <= N+1 matter, but N is 5000, next prime is 5003 which is > N+1? No, 5003 > 5001. So primes <= 5001 are relevant. 
# Actually, if len(S) = k, we check p <= k. Max k = 5001. So we need primes <= 5001.
PRIMES = _primes(5001)
PRIMES_ARRAY = np.array(PRIMES)

def _is_admissible_mask(mask, primes_array):
    """
    Check if the set of indices where mask is True is admissible.
    Returns True if admissible.
    """
    n_selected = np.sum(mask)
    if n_selected == 0:
        return True
    
    # We only need to check primes p <= n_selected
    relevant_primes = primes_array[primes_array <= n_selected]
    
    if len(relevant_primes) == 0:
        return True
        
    # Get indices of selected elements
    # This is the bottleneck if done naively, but we can optimize by precomputing residues?
    # No, residues depend on p.
    # Let's use the bitwise trick.
    # For each p, we want to know if all residues 0..p-1 are present.
    
    # This loop over primes is small (primes <= 5001 is ~669 primes).
    # But doing it for every step in local search might be slow if not vectorized well.
    
    # Alternative: Precompute residue counts?
    # Let's try a faster check for a single candidate addition/removal/swap.
    
    # For now, let's just check full admissibility using vectorized ops where possible.
    # Actually, checking full admissibility is O(P * K) where P is #primes, K is #selected.
    # We need a faster way to check if adding/removing a specific element breaks admissibility.
    
    return True # Placeholder, we will implement specific check functions

def _check_violation_add(mask, x, primes_array):
    """
    Check if adding x to the set represented by mask causes a violation.
    Returns True if adding x causes a violation (i.e., makes tuple inadmissible).
    """
    # New size
    current_size = np.sum(mask)
    new_size = current_size + 1
    
    # Primes to check: p <= new_size
    # A violation occurs if for some p <= new_size, the new set covers all residues mod p.
    # The old set was admissible, so for all p <= current_size, some residue was missing.
    # Adding x can only fill the missing residue for a specific p.
    # So a violation happens if:
    # 1. There exists a prime p <= current_size such that:
    #    a. In the old set, exactly one residue r was missing mod p.
    #    b. x % p == r.
    # 2. OR, there exists a prime p such that current_size < p <= new_size (i.e., p = new_size if prime)
    #    and the new set covers all residues. But if p > current_size, the old set had < p elements, 
    #    so it couldn't cover all residues. The new set has p elements. It covers all residues 
    #    iff all p elements are distinct mod p. This is a strong condition.
    
    # Efficient check:
    # We need to know for each relevant prime p, how many residues are covered by the current mask.
    # If count == p-1 and the missing one is x%p, then violation.
    
    # This requires knowing the missing residues.
    # Let's precompute for the current mask? No, mask changes.
    
    # Let's use a heuristic or a faster specific check.
    # Given the time limit, we might need to be very clever.
    
    # Let's try a different approach: 
    # Instead of checking admissibility from scratch, we maintain the "missing residues" state?
    # That's hard with swaps.
    
    # Let's stick to the simple check but optimize it.
    # For a single x, we can check primes p <= new_size.
    # For each p, we check if (mask residues union {x%p}) == all residues.
    
    # Vectorized check for one x:
    # This is still O(P). P is ~660.
    # If we do 10000 swaps, that's 6.6M operations. In NumPy, this is fast.
    
    selected = np.where(mask)[0]
    if len(selected) == 0:
        # Empty set is admissible. Adding x makes size 1. 
        # Primes p <= 1: none. So always safe to add? 
        # Wait, if size becomes 1, we check p <= 1. No primes. So yes.
        return False

    # Primes <= new_size
    relevant_primes = primes_array[primes_array <= new_size]
    
    # For each prime p, check if all residues are present in (selected + x)
    # We can compute residues of selected mod p, and see if x%p is the only missing one.
    
    # Optimization: 
    # If we precompute the residue counts for the current mask for all primes?
    # That's expensive to update.
    
    # Let's just do the check.
    for p in relevant_primes:
        # Residues of current selected
        res_sel = selected % p
        # Unique residues present
        # If len(unique) == p, it was already a violation (should not happen if mask is valid)
        # If len(unique) == p-1, check if x%p is the missing one.
        
        # Fast check for "is x%p the only missing one?"
        # Create a boolean array of size p
        # present = np.zeros(p, dtype=bool)
        # present[res_sel] = True
        # missing = np.where(~present)[0]
        # if len(missing) == 1 and missing[0] == x % p: return True
        
        # Even faster:
        # Check if x%p is in present. If yes, no violation from this p (unless already violated, which it isn't).
        # If x%p is NOT in present, check if it's the ONLY missing one.
        
        # To check if it's the only missing one, we need to know if all other p-1 residues are present.
        # This is equivalent to: len(np.unique(np.concatenate([res_sel, [x%p]]))) == p
        
        # Let's use a bitset approach if p is small? No, p can be up to 5000.
        
        # Let's just use unique.
        # This might be slow in Python loop.
        
        # Alternative:
        # If we maintain a global "coverage" structure?
        
        # Let's try to vectorize over primes? Hard because p varies.
        
        # Let's assume the Python loop over ~660 primes is acceptable if the inner check is fast.
        # Inner check: 
        # 1. Check if x%p is already in selected residues.
        # 2. If not, check if selected residues cover all other p-1 values.
        
        # 1. is x%p in res_sel?
        #    We can use a set for res_sel? 
        #    Or just check: if np.any(res_sel == x%p): continue
        
        # 2. If not, check if len(np.unique(res_sel)) == p-1.
        #    This is expensive.
        
        # Better: 
        # We know the set is currently admissible. So for each p, at least one residue is missing.
        # We want to know if x fills the LAST missing residue.
        
        # Let's precompute the missing residues for the current mask?
        # No, let's just do the naive check but with early exit.
        
        # Actually, for most primes, the number of missing residues will be > 1.
        # So adding x won't cause a violation.
        # Only for primes where exactly 1 residue is missing, we have a risk.
        
        # How to find primes where exactly 1 residue is missing?
        # This requires counting.
        
        # Let's try a different strategy: 
        # Randomized Local Search with a "Relaxation" phase.
        # Start with a dense set, then remove elements to satisfy constraints.
        # This is the "remove minority" greedy.
        
        # The current program does this. 
        # To improve, we need better tie-breaking or a different heuristic.
        
        # What if we use a "Greedy with Lookahead"?
        # Or "Simulated Annealing"?
        
        # Let's try Simulated Annealing on the mask.
        # Move: Flip a bit (add or remove).
        # Cost: -size (we want to maximize size).
        # Constraint: Admissibility.
        # If a move breaks admissibility, it's rejected (or accepted with low prob if we allow temporary violations? No, must be valid).
        # So we only accept moves that keep the set admissible.
        
        # To make this fast, we need a fast admissibility check for a single flip.
        
        # Let's implement `_check_violation_flip(mask, x, is_add)`.
        
        # For `is_add`:
        #   As above, check if x fills the last missing residue for any p <= new_size.
        
        # For `is_remove`:
        #   Removing x can never cause a violation (it only removes coverage). 
        #   Wait, if we remove x, the set becomes smaller. 
        #   Admissibility condition: for all p <= new_size, some residue missing.
        #   If old set was admissible, removing an element keeps it admissible?
        #   Yes! If a residue was missing before, it's still missing.
        #   If no residue was missing (violation), it was invalid to begin with.
        #   So removing is ALWAYS safe.
        
        # Therefore, we can always remove elements. We only need to check ADDS.
        
        # So the search strategy:
        # 1. Start with a valid set (e.g., from greedy removal).
        # 2. Try to ADD elements.
        # 3. If an add is blocked, try to REMOVE a different element to "make room" (swap)?
        #    If we swap x_out with x_in, we remove x_out (safe) and add x_in (check).
        #    If x_in is blocked by the current set, maybe removing x_out opens a path?
        #    No, removing x_out only increases missing residues, making it EASIER to add x_in.
        #    So if x_in is blocked in S, it is also blocked in S \ {x_out}.
        #    Wait. 
        #    Violation on add x_in: x_in fills the last missing residue r for some p.
        #    If we remove x_out, does it change the missing residues?
        #    If x_out % p != r, then the missing residues for p are unchanged (r is still missing, and maybe others).
        #    If x_out % p == r, then r was NOT missing in S. But we said r was the ONLY missing one.
        #    Contradiction. If r was the only missing one, then x_out % p != r (since x_out is in S).
        #    So removing x_out (where x_out % p != r) leaves r as the only missing one.
        #    So x_in is STILL blocked.
        
        # Conclusion: Swaps do not help unblock an add if the add is blocked by the current set's coverage.
        # The only way to unblock an add x_in is to remove an element x_out such that x_out % p == r (the missing residue)?
        # No, if r is missing, no element in S has residue r. So we can't remove an element with residue r.
        
        # So, if x_in is blocked, it is blocked by the structure of S. 
        # We cannot fix it by removing other elements.
        # We must change S in a way that removes the "threat" of filling the last residue.
        # But the threat is that r is the ONLY missing residue.
        # To make r NOT the only missing residue, we need to remove an element with residue r' (r' != r) such that r' becomes missing too?
        # No, we need to increase the number of missing residues for p.
        # Currently, for prime p, missing = {r}.
        # We want missing to be {r, r', ...}.
        # So we need to remove an element x_out such that x_out % p is the ONLY element with residue x_out % p in S.
        # If we remove it, that residue becomes missing.
        # Then missing becomes {r, x_out%p}.
        # Then adding x_in (which has residue r) is safe, because there are 2 missing residues.
        
        # So, a Swap CAN unblock an add!
        # Condition: 
        # 1. x_in is blocked by prime p because r = x_in%p is the only missing residue.
        # 2. We find an x_out in S such that x_out % p is a "unique" residue in S (i.e., count of x_out%p in S is 1).
        # 3. We remove x_out and add x_in.
        # 4. We must also check that this swap doesn't cause violations for OTHER primes.
        
        # This is a powerful local search move!
        
        # Algorithm:
        # 1. Start with greedy valid set.
        # 2. Loop:
        #    a. Find all "blocked" adds (x not in S such that adding x causes violation).
        #    b. For each blocked x, identify the "blocking" primes.
        #    c. For each blocking prime p, find an x_out in S that is "removable" for p (unique residue).
        #    d. Perform swap (remove x_out, add x).
        #    e. Verify global admissibility (should be fast if we trust the local logic, but better to be safe).
        #    f. If valid, accept.
        
        # This is complex. Let's implement a simpler version first:
        # Just try random swaps and check admissibility.
        # But we need a fast admissibility check.
        
        # Let's implement `_is_admissible_fast(mask)` using the vectorized residue check.
        
    return False # Placeholder

def construct():
    start_time = time.time()
    
    # 1. Generate initial set using Greedy Removal
    # Start with all elements
    mask = np.ones(N + 1, dtype=bool)
    
    # Primes sorted
    primes_list = PRIMES
    
    # Greedy removal
    while True:
        # Find violating prime
        viol_p = None
        current_size = np.sum(mask)
        for p in primes_list:
            if p > current_size:
                break
            # Check if all residues present
            selected = np.where(mask)[0]
            res = selected % p
            if len(np.unique(res)) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
            
        # Remove minority residue class
        p = viol_p
        selected = np.where(mask)[0]
        res = selected % p
        counts = np.bincount(res, minlength=p)
        min_count = np.min(counts)
        # Find residues with min count
        candidates = np.where(counts == min_count)[0]
        # Pick one (random or first)
        r_to_remove = candidates[0]
        
        # Remove all x in S such that x % p == r_to_remove
        mask[(np.where(mask)[0] % p) == r_to_remove] = False
        
        if time.time() - start_time > MAX_TIME - 10:
            break

    best_size = np.sum(mask)
    best_mask = mask.copy()
    
    # 2. Local Search: Try to add elements
    # We will iterate through candidates to add.
    # To be fast, we'll just try random swaps/adds.
    
    # Let's implement the "Swap to Unblock" logic roughly.
    # Actually, let's just try to add random elements not in S.
    # If it fails, try a swap.
    
    # For simplicity and robustness within time limit, let's do a randomized local search
    # that tries to add elements. If blocked, it tries to remove a random element and retry?
    # No, that decreases size.
    
    # Let's stick to the "Swap" idea.
    
    not_in_S = np.where(~mask)[0]
    in_S = np.where(mask)[0]
    
    if len(not_in_S) == 0:
        return np.where(mask)[0].tolist()
        
    # Precompute residues for current set for all primes? 
    # Too memory heavy.
    
    # Let's just do a simple loop.
    attempts = 0
    max_attempts = 500000 # Adjust based on speed
    
    while time.time() - start_time < MAX_TIME - 5:
        attempts += 1
        
        # Pick a random candidate to add
        x_in = not_in_S[np.random.randint(len(not_in_S))]
        
        # Check if adding x_in is safe
        # We need a fast check.
        # Let's use the "missing residues" logic.
        
        # For speed, let's just try it and verify?
        # No, verification is O(P*K).
        
        # Let's implement a fast check for a single add.
        current_size = np.sum(mask)
        new_size = current_size + 1
        
        safe = True
        for p in primes_list:
            if p > new_size:
                break
            
            # Check if x_in fills the last missing residue
            selected = np.where(mask)[0]
            res_sel = selected % p
            
            # Is x_in % p already in res_sel?
            if np.any(res_sel == x_in % p):
                continue
            
            # If not, is it the ONLY missing one?
            # Check if all other p-1 residues are present.
            # This is equivalent to: len(np.unique(res_sel)) == p - 1
            
            # Optimization: 
            # If current_size < p - 1, then it's impossible to have p-1 unique residues.
            # So it's safe.
            if current_size < p - 1:
                continue
            
            # Check unique count
            # This is the expensive part.
            # We can cache unique counts?
            
            # Let's just do it.
            if len(np.unique(res_sel)) == p - 1:
                safe = False
                break
        
        if safe:
            mask[x_in] = True
            in_S = np.append(in_S, x_in)
            not_in_S = np.delete(not_in_S, np.where(not_in_S == x_in)[0])
            best_size = np.sum(mask)
            best_mask = mask.copy()
        else:
            # Try a swap?
            # Find a blocking prime
            # This is complex. Let's just skip for now and rely on the greedy start being good.
            # Or try a random swap: remove random x_out, add x_in.
            # But we don't know if it's safe.
            
            pass
        
    return np.where(best_mask)[0].tolist()

# To make the fast check faster, we can precompute `np.unique` results? 
# No, they change.

# Let's refine the "safe" check to be faster.
# We can precompute the set of residues for each prime for the current mask?
# That's O(P*K) per step. Too slow.

# Alternative: 
# Use the fact that we only need to check primes where the coverage is "tight" (p-1 unique residues).
# Most primes will have many missing residues.
# We can quickly filter primes where `current_size < p - 1` (safe).
# For the rest, we check.

# The current implementation in `construct` does this.
# The bottleneck is `len(np.unique(res_sel))`.
# We can replace it with:
# `np.sum(np.bincount(res_sel, minlength=p) > 0) == p - 1`
# This is faster? Maybe.

# Let's update the check inside construct.

def _is_add_safe(mask, x, primes_array, current_size):
    new_size = current_size + 1
    selected = np.where(mask)[0]
    
    for p in primes_array:
        if p > new_size:
            break
        
        # Quick check: if current_size < p - 1, safe
        if current_size < p - 1:
            continue
            
        res_sel = selected % p
        # Check if x % p is present
        if np.any(res_sel == x % p):
            continue
            
        # Check if all other p-1 residues are present
        # Use bincount
        counts = np.bincount(res_sel, minlength=p)
        # Number of non-zero counts
        if np.sum(counts > 0) == p - 1:
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # 1. Greedy Removal
    mask = np.ones(N + 1, dtype=bool)
    primes_list = PRIMES
    
    while True:
        viol_p = None
        current_size = np.sum(mask)
        for p in primes_list:
            if p > current_size:
                break
            selected = np.where(mask)[0]
            res = selected % p
            if len(np.unique(res)) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
            
        p = viol_p
        selected = np.where(mask)[0]
        res = selected % p
        counts = np.bincount(res, minlength=p)
        min_count = np.min(counts)
        candidates = np.where(counts == min_count)[0]
        r_to_remove = candidates[0]
        
        mask[(np.where(mask)[0] % p) == r_to_remove] = False
        
        if time.time() - start_time > MAX_TIME - 10:
            break

    best_size = np.sum(mask)
    best_mask = mask.copy()
    
    # 2. Local Search
    not_in_S = np.where(~mask)[0]
    
    if len(not_in_S) == 0:
        return np.where(mask)[0].tolist()
        
    while time.time() - start_time < MAX_TIME - 5:
        if len(not_in_S) == 0:
            break
            
        x_in = not_in_S[np.random.randint(len(not_in_S))]
        current_size = np.sum(mask)
        
        if _is_add_safe(mask, x_in, PRIMES_ARRAY, current_size):
            mask[x_in] = True
            not_in_S = np.delete(not_in_S, np.where(not_in_S == x_in)[0])
            best_size = np.sum(mask)
            best_mask = mask.copy()
            
    return np.where(best_mask)[0].tolist()

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
