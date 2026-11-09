import time
import random
import math
import sys

N = 5000
DEADLINE = 300.0

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

# Primes up to N are candidates for constraints.
# In an admissible set of size K, we only care about primes p <= K.
# Since K <= N+1, we consider all primes up to N+1.
PRIMES = _primes(N + 1)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Helper to calculate the size of the set defined by a pattern
    # pattern: dict {prime: excluded_residue}
    # We want to maximize the number of x in [0, N] such that 
    # for all (p, r) in pattern, x % p != r
    
    def evaluate_pattern(pattern):
        # Quick check: if pattern is empty, size is N+1 (but not admissible if N+1 > 1)
        # We must ensure admissibility: for every p <= |S|, some class is missed.
        # By construction, if we include p in pattern, class r is missed.
        # But what if p is NOT in pattern? Then all classes might be present.
        # So the pattern must contain entries for ALL primes p <= |S|.
        
        # This function assumes the pattern is "complete" for the resulting size.
        # However, to evaluate a candidate pattern during search, we might have incomplete patterns.
        # Let's define the search space as: for each prime p, either we exclude a class r, or we "don't care" (but then we must ensure admissibility later).
        
        # Actually, a simpler model:
        # A valid admissible set A is determined by a function f: P -> Z/pZ where P is a set of primes.
        # If p in P, then A misses class f(p).
        # If p not in P, A might cover all classes, which is invalid if p <= |A|.
        
        # Strategy: We search over patterns for primes up to some limit.
        # Let's iterate over primes in increasing order.
        
        # To count survivors efficiently:
        # We can use the inclusion-exclusion principle or simply filter [0, N].
        # Given N=5000, filtering is O(N * |P|). 
        # |P| is small (primes up to ~5000 is ~670, but we only care about p <= |S|).
        # If |S| is large, we care about many primes.
        
        # Optimization:
        # We can precompute residues for all primes? No, N is small.
        
        # Let's just filter.
        # But we need to know |S| to know which primes matter.
        # This creates a dependency loop.
        
        # Approach:
        # 1. Propose a pattern for a set of primes P_candidate.
        # 2. Compute S_candidate = {x in [0, N] : for all p in P_candidate, x % p != pattern[p]}
        # 3. Let K = |S_candidate|.
        # 4. Check admissibility: For every prime p <= K, does S_candidate miss a class mod p?
        #    - If p in P_candidate: Yes, by construction it misses pattern[p].
        #    - If p NOT in P_candidate: We must check if S_candidate covers all classes mod p.
        #      If it does, the set is NOT admissible.
        
        # So, a pattern is valid if:
        # For all p <= |S|:
        #   if p in pattern: OK
        #   if p not in pattern: S misses at least one class mod p.
        
        # To maximize size, we want to minimize exclusions.
        
        pass

    # A more direct approach:
    # The current best is ~1.05 * 5000 = 5250? No, score is value/seed.
    # Seed score 1.0. Value is tuple size.
    # Max size is 5001.
    # If seed returns something smaller, score 1.0 means seed_size.
    # Current program score 1.0527. If seed is 1.0, maybe seed returns a fixed size?
    # "Score: tuple size". "search score is value/seed".
    # If seed program returns size S_seed, and we return S_new, score = S_new / S_seed.
    # We don't know S_seed. But we want to maximize S_new.
    
    # Let's implement a Local Search on the "Exclusion Pattern".
    # State: dict {p: r} for primes p.
    # We start with a greedy pattern:
    # For each prime p in increasing order:
    #   Assume we have a set S defined by previous primes.
    #   If p <= |S|:
    #      Check if S covers all classes mod p.
    #      If yes, we MUST exclude one class. Choose the one that removes the fewest elements from S.
    #   If p > |S|: We don't need to worry about p anymore (unless |S| grows, which it doesn't).
    
    # This greedy construction yields a valid set.
    # Then we try to improve it by "swapping" exclusions or removing exclusions if possible.
    
    def greedy_construct():
        pattern = {}
        # We need to know the set S to make decisions.
        # S is defined by pattern.
        # We can maintain S as a boolean array or list of active indices.
        # N=5000 is small. Boolean array is fine.
        
        active = [True] * (N + 1)
        current_size = N + 1
        
        for p in PRIMES:
            if p > current_size:
                break
            
            # Check if all classes are covered in active set
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            
            # If any count is 0, we are admissible for p. No action needed.
            if min(counts) == 0:
                continue
            
            # All classes covered. We must remove one.
            # Which one to remove?
            # Option 1: Min count (standard greedy)
            # Option 2: Random among min
            # Option 3: Look ahead?
            
            # Let's try a few variations of greedy initialization
            
            # Variation 1: Min count
            min_count = min(counts)
            candidates = [r for r in range(p) if counts[r] == min_count]
            r = random.choice(candidates)
            
            # Remove class r
            removed = 0
            for x in range(N + 1):
                if active[x] and x % p == r:
                    active[x] = False
                    removed += 1
            current_size -= removed
            pattern[p] = r
            
        points = [x for x in range(N + 1) if active[x]]
        return points, pattern

    def local_search(points, pattern):
        # Try to improve the set.
        # Moves:
        # 1. Change the excluded residue for a prime p.
        # 2. Remove an exclusion for a prime p (if it becomes redundant? No, if we remove it, we might cover all classes).
        #    Actually, if we remove exclusion for p, the set grows.
        #    If the new set is still admissible for all q <= new_size, it's a valid improvement.
        #    This is promising! We can try to "relax" constraints.
        
        active = [False] * (N + 1)
        for x in points:
            active[x] = True
        current_size = len(points)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Strategy A: Try to remove an exclusion (Relax)
            # Iterate through primes in pattern
            # If we remove exclusion for p, does the set remain admissible?
            # The set will grow. Let new_size = old_size + count(p, r).
            # We need to check admissibility for all q <= new_size.
            # This is expensive.
            
            # Strategy B: Try to change exclusion for p to a different residue r'.
            # This keeps size roughly same (or changes slightly).
            
            # Let's try Relaxing first, as it directly increases size.
            # We iterate primes in pattern.
            # To avoid O(P^2) checks, we can be smart.
            
            # For each p in pattern:
            #   Let r = pattern[p].
            #   Temporarily add back class r.
            #   New set S' = S U {x in [0,N] : x%p == r and x not in S}
            #   Actually, S is defined by exclusions. If we remove exclusion p, 
            #   S' = {x : for all q != p, x % q != pattern[q]}
            #   We need to check if S' is admissible.
            
            # Efficient check:
            # 1. Calculate size of S'.
            # 2. For each prime q <= |S'|:
            #    Check if S' misses a class mod q.
            
            # This is still heavy.
            
            # Alternative: Simulated Annealing on the pattern?
            # Objective: Maximize size, subject to admissibility.
            # Penalty for non-admissibility.
            
            pass

    # Given the time limit and complexity, let's stick to a robust Multi-Start Greedy 
    # with a specific twist: **Adaptive Residue Selection**.
    # Instead of just min count, we can look at the "cost" of removal in terms of 
    # future constraints.
    
    # However, the most effective structural change mentioned in inspirations is 
    # "Local search polish that exploits critical prime constraints".
    
    # Let's implement:
    # 1. Generate a valid set via Greedy.
    # 2. Identify "Critical Primes": Primes p <= |S| where the number of elements 
    #    in the least-populated class is small.
    # 3. Try to "swap" the excluded class for these primes to a class that 
    #    allows adding back more elements, while maintaining admissibility for 
    #    other primes.
    
    # Actually, a simpler and very effective structural change is:
    # **Reverse Greedy (Additive)**
    # Start with empty set.
    # Add elements one by one? No, too slow.
    
    # **Filtering with Priority**
    # Process primes in decreasing order?
    # If we process large primes first, we remove fewer elements per prime (N/p).
    # But we must satisfy small primes too.
    # If we satisfy large primes first, we might break small prime admissibility?
    # No, admissibility is "some class is missed". Removing elements only helps 
    # to miss classes. It never creates a situation where a class was missed but 
    # now is covered.
    # Wait.
    # If I have a set S. It misses class r mod p.
    # I remove elements from S to get S'.
    # S' is a subset of S.
    # If S missed class r, S' also misses class r (or more).
    # So removing elements NEVER breaks admissibility.
    
    # This is a crucial insight!
    # If a set is admissible, any subset is admissible.
    # Therefore, we can start with the FULL set [0, N] and remove elements.
    # We MUST remove elements to satisfy primes p <= |S_final|.
    # But we don't know |S_final| in advance.
    
    # However, note:
    # If we remove elements, |S| decreases.
    # The condition "for all p <= |S|" becomes WEAKER as |S| decreases.
    # We only need to satisfy primes up to the final size.
    
    # So, if we start with [0, N], and we remove elements until it is admissible,
    # we might remove too much.
    
    # Let's try a different structural approach:
    # **Binary Search on Size + Feasibility Check**
    # Is it possible to have an admissible set of size K?
    # If yes, try larger K.
    
    # How to check feasibility for K?
    # We need to find a subset A of [0, N] with |A|=K such that for all p<=K, A misses a class.
    # This is still hard.
    
    # Let's go back to **Greedy + Local Search (Swap)**
    # The key to beating 1.05 might be in the local search.
    
    def construct_impl():
        best_points = []
        best_size = 0
        
        # Run multiple greedy starts
        for _ in range(50):
            if time.time() > deadline:
                break
                
            active = [True] * (N + 1)
            current_size = N + 1
            pattern = {}
            
            for p in PRIMES:
                if p > current_size:
                    break
                
                counts = [0] * p
                for x in range(N + 1):
                    if active[x]:
                        counts[x % p] += 1
                
                if min(counts) == 0:
                    continue
                
                # Choose residue to remove
                # Use a heuristic: Remove the class that is "least useful" for future primes?
                # Hard to compute.
                # Just pick min count, random tie-break.
                min_c = min(counts)
                cands = [r for r in range(p) if counts[r] == min_c]
                r = random.choice(cands)
                
                for x in range(N + 1):
                    if active[x] and x % p == r:
                        active[x] = False
                        current_size -= 1
                
                pattern[p] = r
            
            points = [x for x in range(N + 1) if active[x]]
            
            if len(points) > best_size:
                best_size = len(points)
                best_points = points
            
            # Local Search: Try to add back elements
            # We have a valid set S.
            # Can we add an element x not in S?
            # If we add x, the set becomes S U {x}.
            # Is S U {x} admissible?
            # We need to check for all p <= |S|+1.
            # For p in pattern: S missed r. S U {x} misses r UNLESS x % p == r.
            #   If x % p == r, then S U {x} covers class r.
            #   We must check if S U {x} covers ALL classes mod p.
            #   If yes, it's invalid for p.
            # For p NOT in pattern:
            #   S might have missed a class. S U {x} might cover it.
            #   We need to check if S U {x} covers all classes mod p.
            
            # This check is O(P * p) for each x.
            # Too slow for all x.
            
            # Optimization:
            # Only try to add x if it doesn't "complete" any prime's coverage.
            
            # Let's try a limited local search:
            # Try to add elements from the "boundary" of the removed classes.
            
            # Actually, let's try **Simulated Annealing on the Pattern** directly.
            # State: pattern dict.
            # Move: Change pattern[p] to a different residue, or add/remove p.
            # Score: Size of resulting set if valid, else -infinity.
            
            pass

        return best_points

    # The above is getting complex. Let's implement a cleaner **Iterative Improvement**
    # 1. Greedy to get S.
    # 2. Identify "slack" in the constraints.
    # 3. Try to relax constraints (remove exclusions) if the set remains admissible.
    
    def try_relax(points, pattern):
        active = [False] * (N + 1)
        for x in points:
            active[x] = True
        size = len(points)
        
        # Try to remove exclusions for primes in pattern
        # Order: Try to remove exclusions for large primes first?
        # Removing exclusion for p adds ~size/p elements.
        
        # To check if removing exclusion for p is valid:
        # New set S' = {x in [0, N] : for all q in pattern, q != p, x % q != pattern[q]}
        # We need to check if S' is admissible for all r <= |S'|.
        
        # This is expensive.
        
        # Alternative: **Randomized Restart with Better Heuristic**
        # Heuristic: When choosing the residue to remove for prime p,
        # prefer residues that are "aligned" with previously removed residues of other primes.
        # This might create larger gaps, allowing smaller primes to be satisfied with less removal?
        # No, we want to MAXIMIZE size.
        
        # Let's try **Min-Conflict Heuristic**
        # When removing class r for p, look at the counts of classes mod q for q > p.
        # If removing r for p causes some class mod q to become 0, that's good (admissibility for q satisfied for free).
        
        pass

    # Given the constraints, I will implement a robust version of the current strategy 
    # but with a **Local Search phase that attempts to add back elements** 
    # using a fast admissibility check.
    
    def fast_admissibility_check(active, size):
        # Check if active set is admissible for all p <= size
        for p in PRIMES:
            if p > size:
                break
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            if min(counts) > 0:
                return False
        return True

    def local_search_add(points, pattern):
        active = [False] * (N + 1)
        for x in points:
            active[x] = True
        size = len(points)
        
        # Candidates to add: elements not in active
        # We prioritize elements that "break" the fewest constraints?
        # Actually, adding an element x might break admissibility for a prime p
        # if x completes the set of residues mod p.
        
        # For each p <= size, let missing_classes_p = {r : count(r) == 0}
        # If we add x, and x % p in missing_classes_p, and count(r) was 0, 
        # then count(r) becomes 1.
        # If missing_classes_p had size 1, and we fill it, then all classes are covered -> INVALID.
        # So we must avoid adding x if it fills the last missing class for any p <= size.
        
        # Precompute missing classes for current set
        # This is O(P * N) which is fine.
        
        missing = {}
        for p in PRIMES:
            if p > size:
                break
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            miss = [r for r in range(p) if counts[r] == 0]
            missing[p] = miss
            
        # Now, for each candidate x not in active:
        # Check if adding x is safe.
        # x is safe if for all p <= size:
        #   if x % p in missing[p]:
        #       len(missing[p]) > 1  # i.e., there are other missing classes, so filling one doesn't cover all
        #   else:
        #       # x % p is already covered, so adding x doesn't change missing status
        #       pass
        
        # Wait, if x % p is already covered, adding x doesn't make it missing.
        # If x % p is missing, adding x makes it covered.
        # We fail if it was the ONLY missing class.
        
        # So condition: For all p <= size, if x % p in missing[p], then len(missing[p]) > 1.
        
        # We can iterate through candidates and add them greedily if safe.
        
        added = 0
        # Sort candidates? Maybe random order.
        candidates = [x for x in range(N + 1) if not active[x]]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check safety
            safe = True
            for p in PRIMES:
                if p > size:
                    break
                r = x % p
                if r in missing[p]:
                    if len(missing[p]) == 1:
                        safe = False
                        break
            
            if safe:
                active[x] = True
                size += 1
                added += 1
                # Update missing dict?
                # This is expensive.
                # Since N is small, we can just re-check or update incrementally.
                # For simplicity, let's just re-run the check for the next candidate?
                # No, let's update missing dict.
                for p in PRIMES:
                    if p > size:
                        break
                    r = x % p
                    if r in missing[p]:
                        missing[p].remove(r)
                    # Note: if missing[p] becomes empty, we are in trouble for future additions,
                    # but we already checked safety for THIS x.
                    # If missing[p] is now empty, any future x' with x'%p == r will be unsafe?
                    # No, if missing[p] is empty, it means all classes are covered.
                    # Then the set is NOT admissible for p!
                    # Wait.
                    # If missing[p] is empty, it means all classes are covered.
                    # This is INVALID.
                    # My safety check prevents this:
                    # "if r in missing[p] and len(missing[p]) == 1: safe = False"
                    # So if len(missing[p]) == 1, we don't add x if x fills it.
                    # So missing[p] can NEVER become empty during this process?
                    # Initially, missing[p] has size >= 1 (admissible).
                    # We only add x if it doesn't fill the last gap.
                    # So missing[p] size stays >= 1.
                    # Correct.
        
        return [x for x in range(N + 1) if active[x]], added

    # Main Loop
    best_size = 0
    best_points = []
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # 1. Greedy
        active = [True] * (N + 1)
        current_size = N + 1
        pattern = {}
        
        for p in PRIMES:
            if p > current_size:
                break
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            if min(counts) == 0:
                continue
            min_c = min(counts)
            cands = [r for r in range(p) if counts[r] == min_c]
            r = random.choice(cands)
            for x in range(N + 1):
                if active[x] and x % p == r:
                    active[x] = False
                    current_size -= 1
            pattern[p] = r
        
        points = [x for x in range(N + 1) if active[x]]
        
        # 2. Local Search (Add back)
        # We need to pass active array or reconstruct it.
        # Let's reconstruct active for local search to avoid side effects.
        active_ls = [False] * (N + 1)
        for x in points:
            active_ls[x] = True
        
        new_points, _ = local_search_add(points, pattern)
        
        if len(new_points) > best_size:
            best_size = len(new_points)
            best_points = new_points

    return best_points

# Entry point
def construct():
    return construct_impl()
