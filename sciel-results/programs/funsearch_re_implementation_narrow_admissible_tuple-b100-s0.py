import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    # Only need to check primes p <= k
    # We have a precomputed list of primes up to N, filter those <= k
    # Optimization: precompute primes up to max possible size (5000)
    
    for p in primes_list:
        if p > k:
            break
        
        # Check if all residues 0..p-1 are covered
        residues = set()
        # Early exit if we find a missing residue
        # But we need to know if ANY is missing. 
        # It's faster to count or use a boolean array if p is small.
        # p is small (primes <= 5000, but k is likely < 5000, so p is small).
        
        # Using a set is fine for small p.
        # For speed, we can break early if we've seen all p residues.
        seen = [False] * p
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = True
                count += 1
                if count == p:
                    # All residues covered, not admissible for this p
                    return False
        # If loop finishes, not all residues covered, so admissible for this p
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Strategy: Randomized Greedy Repair
    # We perform multiple restarts with different random seeds/choices
    # to escape local minima of the "remove smallest class" heuristic.
    
    num_restarts = 200  # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Start with all numbers
        current_set = set(range(N))
        
        # Greedy Repair Loop
        # We want to make it admissible by removing elements.
        # To maximize size, we remove the "cheapest" constraint violation.
        
        while True:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime
            violating_p = None
            for p in primes:
                if p > k:
                    break
                # Check if all residues mod p are covered
                seen = [False] * p
                count = 0
                for v in current_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = True
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                # Admissible
                break
            
            # We need to remove elements to break the coverage for violating_p
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            # Find the class with the minimum number of elements
            # Removing this class minimizes the loss in size
            min_count = float('inf')
            min_idx = -1
            # Add slight randomization to avoid deterministic traps
            # Pick from the smallest few classes
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            # Randomly pick from the top 3 smallest, or just the smallest
            if random.random() < 0.7:
                min_idx = class_sizes[0][1]
            else:
                # Pick randomly from the 3 smallest
                top_k = min(3, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            # Remove all elements in this class
            to_remove = set(classes[min_idx])
            current_set -= to_remove
            
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        # Try to add back elements to improve size
        # This is a local search: try adding any element from [0, N) not in current_set
        # If it remains admissible, keep it.
        
        # To be efficient, we don't try ALL 5000 elements every step.
        # We can try a random subset or the elements removed previously.
        # However, checking admissibility is O(K * P) where P is number of primes <= K.
        # K ~ 100-200, P ~ 30-50. K*P ~ 10,000 ops.
        # 5000 elements * 10,000 ops = 50,000,000 ops per full pass.
        # This might be too slow for 300s if we do many passes.
        
        # Optimization: Only check primes p <= len(current_set) + 1.
        # And we can stop checking as soon as a violation is found.
        
        # Let's do a few passes of random additions
        max_add_passes = 5
        for _ in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
            # Get a random permutation of elements not in current_set
            # To save time, we can just pick a random sample
            # Or iterate through all, but break early if no additions
            # Let's iterate through all, but only check admissibility if adding x
            # might not break constraints.
            
            # Actually, a simpler heuristic:
            # Try to add elements that were removed in the repair phase.
            # We didn't store them. Let's just try random elements.
            
            # Efficient check:
            # We maintain the current set.
            # For each candidate x not in set:
            #   Check if set U {x} is admissible.
            
            # To speed up, we can pre-compute residue counts for current_set
            # for all relevant primes.
            
            # Let's implement a faster local search step.
            # We'll try to add random elements.
            
            candidates = [x for x in range(N) if x not in current_set]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid timeout
            check_limit = 500
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 1:
                    break
                
                # Check if current_set U {x} is admissible
                new_size = len(current_set) + 1
                is_adm = True
                for p in primes:
                    if p > new_size:
                        break
                    # Check if all residues mod p are covered in current_set U {x}
                    # We can count residues in current_set, then see if x fills the last gap
                    # This is complex. Simple check is safer.
                    
                    # Optimization: 
                    # If current_set already covers all residues mod p, it's already invalid?
                    # No, current_set IS admissible, so it misses at least one residue for each p.
                    # Adding x might cover the missing residue, making it invalid.
                    # So we only need to check if x covers the *only* missing residue for some p.
                    
                    # Let's just do the full check for simplicity, it's fast enough for small K.
                    seen = [False] * p
                    count = 0
                    # Iterate current_set
                    for v in current_set:
                        r = v % p
                        if not seen[r]:
                            seen[r] = True
                            count += 1
                            if count == p:
                                # This shouldn't happen if current_set is admissible
                                # Wait, if current_set is admissible, it MUST miss at least one residue.
                                # So count < p is guaranteed for current_set.
                                pass
                    # Now check x
                    rx = x % p
                    if not seen[rx]:
                        count += 1
                        if count == p:
                            is_adm = False
                            break
                if is_adm:
                    current_set.add(x)
            
            # If we added some, we might be able to add more in next pass
            # If we added none, we are stuck? Not necessarily, but likely local max.
            # We can break early if no additions were made in this pass.
            # We don't track additions count here easily.
            # Let's just do a fixed number of passes.

        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return list(best_set)

# Initial call to set start time
construct()
