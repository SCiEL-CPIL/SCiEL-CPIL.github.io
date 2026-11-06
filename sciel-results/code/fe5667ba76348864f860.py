import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin for 150s limit
    
    # 1. Generate Primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    all_primes = get_primes(1000)
    
    # 2. Select Prime Set
    # Use primes >= 11, remove largest if sum > 1
    candidates = [p for p in all_primes if p >= 11]
    candidates_sorted_desc = sorted(candidates, reverse=True)
    
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    
    while current_sum > 1:
        if not candidates_sorted_desc:
            break
        p_to_remove = candidates_sorted_desc.pop()
        primes_set.remove(p_to_remove)
        current_sum -= Fraction(1, p_to_remove)
        
    primes_set = sorted(primes_set)
    n_primes = len(primes_set)
    
    # CSP-based greedy set cover with forward checking
    # Key insight: instead of local search on residues, greedily assign each uncovered
    # integer to the prime that covers the most new integers (maximal gain),
    # with forward checking to ensure we don't paint ourselves into a corner.
    
    L_CHECK_LIMIT = 5000
    
    def greedy_cover(L_limit):
        """
        Greedy set cover: iteratively pick the (p, r) that covers the most uncovered integers.
        This is a structural change from local search - we build the solution from scratch
        using a set-cover heuristic with forward checking.
        """
        # Track which integers are covered
        covered = bytearray(L_limit + 1)  # 0 = uncovered, 1 = covered
        # Track which primes are used
        used_primes = set()
        assignment = {}  # p -> r
        
        # For each prime, precompute the coverage sets for each residue
        # But that's too expensive. Instead, we'll compute on the fly.
        
        # Precompute for each prime p, the list of integers in 1..L_limit for each residue
        # Actually, let's just iterate and compute coverage incrementally.
        
        # We need to be careful: once we assign residue r to prime p, that's fixed.
        # The constraint is that we can only use each prime once.
        
        # Greedy approach:
        # 1. Start with no assignments
        # 2. At each step, find the (p, r) pair that covers the most currently-uncovered integers
        # 3. Assign it, mark those integers as covered
        # 4. Repeat until no more uncovered integers or no more primes available
        
        # To make this efficient, we need to quickly compute for each (p, r) how many
        # new integers it would cover.
        
        # For prime p with residue r, it covers: r, r+p, r+2p, ... (within 1..L_limit)
        # where r is in [0, p-1] and we interpret r=0 as covering p, 2p, 3p, ...
        
        # Let's precompute for each prime p, the count of uncovered integers for each residue.
        # This is O(L_limit / p) per residue, so O(L_limit) per prime, O(n_primes * L_limit) total.
        # For n_primes ~ 140 and L_limit = 5000, that's ~700K operations, which is fine.
        
        best_L = 0
        best_classes = None
        
        # We'll do multiple passes with different strategies
        for pass_num in range(3):
            covered = bytearray(L_limit + 1)
            used_primes = set()
            assignment = {}
            
            # Count uncovered
            uncovered_count = L_limit
            
            while uncovered_count > 0 and len(used_primes) < n_primes:
                # Find the best (p, r) that covers the most uncovered integers
                best_gain = 0
                best_p = -1
                best_r = -1
                
                for p in primes_set:
                    if p in used_primes:
                        continue
                    
                    # For this prime, try each residue and count uncovered coverage
                    # We can do this efficiently by checking which multiples are uncovered
                    for r in range(p):
                        # Count how many uncovered integers are in {r, r+p, r+2p, ...} ∩ [1, L_limit]
                        # where r=0 means {p, 2p, 3p, ...}
                        start = r if r > 0 else p
                        if start > L_limit:
                            gain = 0
                        else:
                            gain = 0
                            for n in range(start, L_limit + 1, p):
                                if not covered[n]:
                                    gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                            best_r = r
                
                if best_p == -1 or best_gain == 0:
                    break
                
                # Assign best_p -> best_r
                used_primes.add(best_p)
                assignment[best_p] = best_r
                
                # Update covered
                start = best_r if best_r > 0 else best_p
                for n in range(start, L_limit + 1, best_p):
                    if not covered[n]:
                        covered[n] = 1
                        uncovered_count -= 1
            
            # Compute L
            L = 0
            for n in range(1, L_limit + 1):
                if covered[n]:
                    L = n
                else:
                    break
            
            if L > best_L:
                best_L = L
                best_classes = [[p, assignment.get(p, 0)] for p in primes_set if p in assignment]
            
            if time.time() - start_time > DEADLINE:
                break
        
        return best_L, best_classes
    
    # Run greedy cover
    best_L, best_classes = greedy_cover(L_CHECK_LIMIT)
    
    # If we have time, try a second phase with larger L
    if best_L >= L_CHECK_LIMIT * 0.9 and time.time() - start_time < DEADLINE - 30:
        L_CHECK_LIMIT_2 = 20000
        L2, classes2 = greedy_cover(L_CHECK_LIMIT_2)
        if L2 > best_L:
            best_L = L2
            best_classes = classes2
    
    # If we have time, try local search refinement on the best solution found
    if best_classes is not None and time.time() - start_time < DEADLINE - 10:
        # Convert to residue array for local search
        residue_map = {p: r for p, r in best_classes}
        residues = [residue_map.get(p, 0) for p in primes_set]
        
        L_limit = L_CHECK_LIMIT
        cover_count = [0] * (L_limit + 1)
        for i, p in enumerate(primes_set):
            r = residues[i]
            start = r if r > 0 else p
            if start <= L_limit:
                for n in range(start, L_limit + 1, p):
                    cover_count[n] += 1
        
        def calculate_L(cover_count, L_limit):
            L = 0
            for n in range(1, L_limit + 1):
                if cover_count[n] == 0:
                    break
                L = n
            return L
        
        L = calculate_L(cover_count, L_limit)
        
        # Local search refinement
        stuck_counter = 0
        max_stuck = 500
        while time.time() - start_time < DEADLINE - 5:
            if L >= L_limit:
                break
            
            hole = L + 1
            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for idx in indices:
                p = primes_set[idx]
                r_old = residues[idx]
                r_new = hole % p
                
                if r_old == r_new:
                    continue
                
                # Apply change
                start_old = r_old if r_old > 0 else p
                if start_old <= L_limit:
                    for n in range(start_old, L_limit + 1, p):
                        cover_count[n] -= 1
                
                start_new = r_new if r_new > 0 else p
                if start_new <= L_limit:
                    for n in range(start_new, L_limit + 1, p):
                        cover_count[n] += 1
                
                residues[idx] = r_new
                new_L = calculate_L(cover_count, L_limit)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    # Revert
                    if start_new <= L_limit:
                        for n in range(start_new, L_limit + 1, p):
                            cover_count[n] -= 1
                    if start_old <= L_limit:
                        for n in range(start_old, L_limit + 1, p):
                            cover_count[n] += 1
                    residues[idx] = r_old
                else:
                    # Revert
                    if start_new <= L_limit:
                        for n in range(start_new, L_limit + 1, p):
                            cover_count[n] -= 1
                    if start_old <= L_limit:
                        for n in range(start_old, L_limit + 1, p):
                            cover_count[n] += 1
                    residues[idx] = r_old
            
            if best_idx != -1:
                p = primes_set[best_idx]
                r_old = residues[best_idx]
                r_new = best_r_new
                
                start_old = r_old if r_old > 0 else p
                if start_old <= L_limit:
                    for n in range(start_old, L_limit + 1, p):
                        cover_count[n] -= 1
                start_new = r_new if r_new > 0 else p
                if start_new <= L_limit:
                    for n in range(start_new, L_limit + 1, p):
                        cover_count[n] += 1
                
                residues[best_idx] = r_new
                L = best_new_L
                stuck_counter = 0
                
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    break
    
    if best_classes is None:
        # Fallback
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_set)]

    return best_classes

# EVOLVE-BLOCK-END
