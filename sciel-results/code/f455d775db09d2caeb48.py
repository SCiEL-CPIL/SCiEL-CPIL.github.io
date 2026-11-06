import time
import random
import math
import numpy as np

N = 5000
DEADLINE_MARGIN = 5.0
T_START = time.time()

# Precompute primes up to N+10
_MAX_PRIME_LIMIT = N + 10
_primes_cache = {}

def _get_primes(limit):
    if limit in _primes_cache:
        return _primes_cache[limit]
    if limit > 100 and limit <= _MAX_PRIME_LIMIT:
        if _MAX_PRIME_LIMIT not in _primes_cache:
            sieve = bytearray([1]) * (_MAX_PRIME_LIMIT + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(_MAX_PRIME_LIMIT ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : _MAX_PRIME_LIMIT + 1 : p] = bytearray(len(sieve[p * p : _MAX_PRIME_LIMIT + 1 : p]))
            _primes_cache[_MAX_PRIME_LIMIT] = [i for i in range(2, _MAX_PRIME_LIMIT + 1) if sieve[i]]
        res = [p for p in _primes_cache[_MAX_PRIME_LIMIT] if p <= limit]
        _primes_cache[limit] = res
        return res
    else:
        if limit < 2:
            res = []
        else:
            sieve = bytearray([1]) * (limit + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(limit ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : limit + 1 : p] = bytearray(len(sieve[p * p : limit + 1 : p]))
            res = [i for i in range(2, limit + 1) if sieve[i]]
        _primes_cache[limit] = res
        return res

_all_primes = _get_primes(_MAX_PRIME_LIMIT)

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _check_admissible(points):
    if not points:
        return True
    m = len(points)
    primes = _get_primes(m)
    for p in primes:
        seen = [0] * p
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True

class ConflictDrivenSearch:
    def __init__(self, N):
        self.N = N
        self.primes = [p for p in _all_primes if p <= N + 1]
        # For fast lookup of which elements are in which residue class for each prime
        # We will maintain this dynamically
        
    def repair_and_expand(self, initial_set, max_time):
        """
        Starts with a set, repairs it by deleting elements to resolve conflicts,
        then expands it by adding elements.
        """
        S = set(initial_set)
        # Data structure for fast conflict checking
        # filled[p][r] = count of elements in S with x % p == r
        filled = {}
        for p in self.primes:
            filled[p] = [0] * p
            
        for x in S:
            for p in self.primes:
                r = x % p
                filled[p][r] += 1
                
        # Helper to check if a prime p is "full" (no empty residue)
        # We only care about p <= len(S)
        
        def is_conflict(p, size):
            if p > size:
                return False
            # Check if any residue is 0
            # Optimization: maintain count of zero residues
            # For now, just check min
            return min(filled[p]) > 0

        # Repair Phase: Delete elements to remove conflicts
        # We prioritize deleting elements from "dense" residue classes of conflicting primes
        while _time_left() > 2.0:
            size = len(S)
            # Find conflicts
            conflicts = [p for p in self.primes if p <= size and is_conflict(p, size)]
            if not conflicts:
                break
            
            # Pick a random conflict
            p = random.choice(conflicts)
            
            # Pick a residue class that is "most full" or just random?
            # To maximize size retention, we should remove an element that
            # helps resolve the most conflicts? 
            # Simple heuristic: Pick a random element from the "most populated" residue class of p
            # Actually, removing ANY element from a full prime p makes it not full?
            # No, if all residues are >= 1, removing one makes one residue possibly 0?
            # If we remove x, filled[p][x%p] decreases.
            # If filled[p][x%p] was 1, it becomes 0 -> conflict resolved for p.
            # If filled[p][x%p] was > 1, it stays > 0 -> conflict remains for p.
            # So we MUST remove an element from a residue class that currently has count 1.
            
            # Find residues with count 1
            candidates_res = [r for r in range(p) if filled[p][r] == 1]
            if not candidates_res:
                # This shouldn't happen if is_conflict is true (min > 0 implies all >= 1)
                # But if min > 1, removing one doesn't help. 
                # Wait, if min > 1, then NO single removal resolves the conflict for p.
                # We need to remove enough elements to drop some residue to 0.
                # This is a "hard" conflict. 
                # Strategy: Pick a random residue r, and remove elements from it until it hits 0?
                # Or just pick a random element and hope it helps other primes?
                # Let's pick a random residue with the smallest count.
                min_count = min(filled[p])
                candidates_res = [r for r in range(p) if filled[p][r] == min_count]
            
            r = random.choice(candidates_res)
            
            # Find an element in S with x % p == r
            # We need a fast way to find this. 
            # We can scan S. S is at most 5000.
            # To make it faster, we could maintain a list of elements per (p, r)? 
            # That's 600 primes * avg 100 residues * elements. Too much memory/update cost.
            # Scanning 5000 elements is ~5000 ops. 
            # If we do 1000 repairs, that's 5M ops. Acceptable.
            
            # Find first x in S such that x % p == r
            x_to_remove = None
            for x in S:
                if x % p == r:
                    x_to_remove = x
                    break
                    
            if x_to_remove is None:
                # Should not happen
                continue
                
            # Remove x_to_remove
            S.remove(x_to_remove)
            for p2 in self.primes:
                r2 = x_to_remove % p2
                filled[p2][r2] -= 1
                
        # Expansion Phase: Try to add elements
        # We want to add elements that don't create new conflicts.
        # A new element x creates a conflict for prime p <= new_size if:
        # 1. p was not conflicting before (had an empty residue)
        # 2. x falls into the LAST empty residue for p (making it full)
        # 3. AND p <= new_size
        
        # We'll iterate through candidates in random order
        candidates = [x for x in range(self.N + 1) if x not in S]
        random.shuffle(candidates)
        
        # We can only add if we don't break admissibility.
        # To be safe, we check admissibility after each add? 
        # That's O(P * S) per add. Too slow.
        
        # Fast check:
        # Adding x affects primes p where x % p == r.
        # It only creates a conflict if:
        # - p <= len(S) + 1
        # - filled[p][r] was 0 (it's the last empty one)
        # - AND there are no other empty residues for p (i.e., this was the ONLY empty one)
        
        for x in candidates:
            if _time_left() < 2.0:
                break
            new_size = len(S) + 1
            safe = True
            for p in self.primes:
                if p > new_size:
                    break
                r = x % p
                if filled[p][r] == 0:
                    # This residue was empty.
                    # Is it the LAST empty one?
                    # If yes, adding x makes it full -> conflict (if p <= new_size)
                    # We need to check if there are any OTHER empty residues.
                    # If there are other empty residues, adding x is safe.
                    # If this is the only empty residue, adding x is unsafe.
                    
                    # Count empty residues for p
                    # We can maintain a count of zero residues? 
                    # Let's just check min(filled[p]) == 0 and sum(filled[p]==0) == 1?
                    # That's O(p).
                    
                    # Optimization: Precompute zero_count for each p?
                    # Let's just do the check.
                    zero_count = 0
                    for count in filled[p]:
                        if count == 0:
                            zero_count += 1
                            if zero_count > 1:
                                break
                    if zero_count == 1:
                        # This is the only empty one. Adding x fills it.
                        # Conflict!
                        safe = False
                        break
            
            if safe:
                S.add(x)
                for p in self.primes:
                    r = x % p
                    filled[p][r] += 1

        return S

def construct():
    # 1. Get baseline from a simple greedy repair (fast)
    # Let's use the seed's repair as a baseline if it's fast enough
    # But we want a structurally different approach.
    # We will run ConflictDrivenSearch multiple times with different initial sets.
    
    best_size = 0
    best_set = []
    
    # Strategy: 
    # Start with a random set of size M, where M is slightly larger than expected best.
    # The expected best is around 500-1000? 
    # Let's estimate. 
    # If we pick 5000 numbers, we cover all residues for small primes.
    # We need to remove enough to leave holes.
    # For p=2, we need to remove all evens or all odds? No, just one residue class.
    # If we remove ~N/p elements from each residue class?
    # Actually, to leave a hole in residue r mod p, we must remove ALL elements in that residue class.
    # So for each prime p, we must remove ~N/p elements.
    # Sum of N/p for p <= K.
    # If K=1000, sum N/p ~ N * log(log(K)) ~ 5000 * 2 = 10000? 
    # Wait, the constraints are coupled.
    # The size is likely much smaller than 5000.
    # Let's start with M = 2000, 3000, etc.
    
    search = ConflictDrivenSearch(N)
    
    # Run multiple trials with different initial sizes
    # We'll try sizes from 1500 to 3500
    # And random subsets.
    
    # To be efficient, we'll run a loop of trials.
    # Each trial:
    # 1. Pick a target size M.
    # 2. Pick random M elements from [0, N].
    # 3. Run repair_and_expand.
    # 4. Update best.
    
    # We'll allocate time based on _time_left.
    
    trials = 0
    while _time_left() > 10.0:
        trials += 1
        # Vary the initial size
        # Start with larger, go smaller?
        # Or random?
        # Let's try a few specific sizes.
        # The current best is ~1.0048 * seed_size.
        # What is seed_size?
        # The seed program's score is 1.0048.
        # If seed_size is, say, 400, then best is ~402.
        # If seed_size is 1000, best is ~1005.
        # Let's assume the optimal size is in the hundreds.
        
        # Let's try M = 1000, 1500, 2000.
        # We'll cycle through them.
        M = 1000 + (trials % 5) * 200 # 1000, 1200, 1400, 1600, 1800
        
        # Cap M at N
        M = min(M, N)
        
        # Pick random subset
        # Using random.sample is O(M log N) or O(M) if we use a set?
        # random.sample(range(N+1), M) is efficient.
        initial_set = set(random.sample(range(N + 1), M))
        
        # Run repair
        # We need to be careful with time. 
        # repair_and_expand has an internal time check.
        # We should give it a time budget.
        # Let's just let it run until _time_left is low.
        
        # To avoid one trial eating all time, we can pass a max_time?
        # The current implementation of repair_and_expand checks _time_left globally.
        # This is dangerous if one trial is slow.
        # Let's add a local time check in the loop inside repair_and_expand?
        # It already has `while _time_left() > 2.0`.
        # This is fine.
        
        try:
            result_set = search.repair_and_expand(initial_set, _time_left())
            if len(result_set) > best_size:
                best_size = len(result_set)
                best_set = list(result_set)
        except Exception as e:
            pass

    # Fallback: if no set found, return empty or a small valid set
    if not best_set:
        # Try a very small set, e.g., {0}
        best_set = [0]
        best_size = 1

    # Final check
    if _check_admissible(best_set):
        return best_set
    else:
        # Should not happen, but just in case
        return []

# For debugging/testing purposes, we might want to ensure the primes are computed
# at import time if needed, but _get_primes is lazy.

if __name__ == "__main__":
    # This block is for local testing, not used by the evaluator
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end - start:.2f}s")
