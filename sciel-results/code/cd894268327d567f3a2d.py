import time
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Helper: Verify a set of classes covers 1..L and respects budget
    def verify_classes(classes, L):
        if not classes:
            return L <= 0
        
        # Check distinct primes and valid residues
        primes_used = set()
        for p, r in classes:
            if p in primes_used:
                return False
            primes_used.add(p)
            if r < 0 or r >= p:
                return False
        
        # Check budget using exact rationals
        s = Fraction(0)
        for p, r in classes:
            s += Fraction(1, p)
        if s > 1:
            return False
        
        # Check coverage
        covered = np.zeros(L + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= L:
                idx = np.arange(start, L + 1, p)
                covered[idx] = True
        
        return np.all(covered[1:])

    # Helper: Calculate the coverage length L for a given set of classes
    # Returns the largest L such that 1..L are all covered.
    def calc_coverage_length(classes, max_check=100000):
        if not classes:
            return 0
        
        # We will check up to max_check.
        # If 1..max_check are covered, we return max_check (or extend if needed, but capped by task to 10^5)
        
        covered = np.zeros(max_check + 1, dtype=bool)
        
        for p, r in classes:
            start = p if r == 0 else r
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p)
                covered[idx] = True
        
        # Find first gap
        # We can use np.where on the slice 1..max_check
        # ~covered[1:] gives True where not covered
        uncovered = np.where(~covered[1:])[0]
        
        if len(uncovered) == 0:
            # All 1..max_check covered
            return max_check
        else:
            return uncovered[0]

    # Helper: Greedy assignment of residues for a fixed set of primes
    # Returns list of [p, r]
    def greedy_assign(primes_set, max_check=10000):
        primes_sorted = sorted(primes_set)
        residues = {}
        
        covered = np.zeros(max_check + 1, dtype=bool)
        
        for p in primes_sorted:
            best_r = 0
            best_gain = -1
            
            for r in range(p):
                start = p if r == 0 else r
                if start > max_check:
                    continue
                idx = np.arange(start, max_check + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[p] = best_r
            start = p if best_r == 0 else best_r
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p)
                covered[idx] = True
        
        return [[p, residues[p]] for p in primes_sorted]

    # Improved Local Search: Targeted gap-filling
    def local_search_targeted(classes, max_iter=500, max_check=100000):
        current_classes = [list(c) for c in classes]
        current_L = calc_coverage_length(current_classes, max_check)
        
        if current_L == 0:
            return current_classes, 0
            
        # Precompute indices for each prime to speed up updates?
        # Or just recompute coverage around the gap.
        
        # We maintain the current coverage array to avoid full recalc if possible?
        # But changing residue changes the whole pattern for that prime.
        # However, we only care about the gap.
        
        # Strategy:
        # 1. Find first gap G.
        # 2. For each prime p, try setting r = G % p.
        # 3. Check if this fixes G and extends L.
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            # Find current gap
            # We know 1..current_L are covered.
            # Check current_L + 1
            # Actually, calc_coverage_length gives us the first gap.
            # Let's just use the returned L. The gap is L+1.
            G = current_L + 1
            
            if G > max_check:
                break
                
            best_new_L = current_L
            best_move = None # (index, new_r)
            
            # Try each prime
            for i, (p, r_old) in enumerate(current_classes):
                r_new = G % p
                
                if r_new == r_old:
                    continue
                
                # Apply move
                old_r = current_classes[i][1]
                current_classes[i][1] = r_new
                
                # Check if 1..G are covered
                # We don't need to check 1..G-1 if we assume they were covered and 
                # changing p doesn't uncover them? 
                # Wait, changing p's residue *can* uncover numbers < G.
                # Example: p=5, r=0 covers 5,10,15. r=1 covers 1,6,11.
                # If G=6, and we had r=0, 6 was not covered by p=5. 
                # If we change to r=1, 6 is covered. 
                # But did we uncover 5? Yes, 5 is no longer covered by p=5.
                # If no other prime covers 5, we break coverage at 5.
                # So we MUST verify 1..G are covered.
                
                # Efficient check:
                # Build coverage array for 1..G? 
                # Since G is small (starts small, grows), this is fast.
                # But G can grow to 10^5. 
                # Building an array of size G for every candidate move is O(N_primes * G).
                # N_primes ~ 100. G ~ 1000. 100*1000 = 10^5 ops. 
                # 500 iters * 100 primes * 10^5 = 5*10^9 ops. Too slow.
                
                # Optimization:
                # Only check if the move *could* uncover a number < G.
                # The numbers covered by p with r_old are $k \cdot p + r_{old}$.
                # The numbers covered by p with r_new are $k \cdot p + r_{new}$.
                # The difference is small.
                # Actually, let's just do a fast check for 1..G.
                # We can precompute a "coverage count" array? 
                # If we maintain `counts` array where counts[x] is number of classes covering x.
                # Then updating a class is O(G/p) ~ O(G).
                # Total update time for all primes in one pass: Sum(G/p) ~ G log log G.
                # This is much faster.
                
                pass # Placeholder for optimized logic below
                
            # Since the above placeholder is complex to implement inline without restructuring,
            # let's stick to a simpler but effective heuristic:
            # Randomized Local Search with full recalculation but limited scope.
            
            # Actually, the previous `local_search` in the code was slow because it tried ALL residues.
            # This targeted one only tries 1 specific residue per prime (the one that covers G).
            # This reduces candidates from Sum(p) to N_primes.
            # N_primes ~ 100.
            # Recalculation cost: O(N_primes * G).
            # If G is small, this is fast.
            
            # Let's implement the targeted search with a "safe" check.
            
            for i, (p, r_old) in enumerate(current_classes):
                r_new = G % p
                if r_new == r_old:
                    continue
                
                # Temporarily apply
                current_classes[i][1] = r_new
                
                # Check coverage 1..G
                # We can do this by checking if any x in 1..G is uncovered.
                # To speed this up, we can just check the "risky" numbers.
                # But for correctness, let's use a small helper.
                
                # Helper to check if 1..L_check are covered by classes
                def is_covered_up_to(classes, L_check):
                    # Only check up to L_check
                    # Use a set of uncovered numbers?
                    # Or boolean array.
                    # Since L_check is the current gap, it's the smallest uncovered.
                    # If we change a class, we might uncover something smaller.
                    # So we must check 1..L_check.
                    
                    # Optimization: If L_check is small, boolean array is fine.
                    # If L_check is large, we might be slow.
                    # But G starts small.
                    
                    covered = np.zeros(L_check + 1, dtype=bool)
                    for p_c, r_c in classes:
                        start = p_c if r_c == 0 else r_c
                        if start <= L_check:
                            idx = np.arange(start, L_check + 1, p_c)
                            covered[idx] = True
                    return np.all(covered[1:])
                
                if is_covered_up_to(current_classes, G):
                    # G is covered. Now see how far we can go.
                    new_L = calc_coverage_length(current_classes, max_check)
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_move = (i, r_new)
                    else:
                        # Revert if no improvement
                        current_classes[i][1] = r_old
                else:
                    # Revert
                    current_classes[i][1] = r_old
            
            if best_move:
                # Apply the best move
                idx, new_r = best_move
                current_classes[idx][1] = new_r
                current_L = best_new_L
                improved = True
                
        return current_classes, current_L

    # Main Strategy:
    # 1. Generate candidate sets.
    # 2. Greedy + Targeted Local Search.
    # 3. Random Restarts.
    
    best_classes = []
    best_L = 0
    
    # Candidate generation
    def smart_trim(prime_list):
        sorted_asc = sorted(prime_list)
        result = []
        current_sum = Fraction(0)
        for p in sorted_asc:
            if current_sum + Fraction(1, p) <= 1:
                result.append(p)
                current_sum += Fraction(1, p)
        return result
    
    base_sets = [
        [p for p in primes if p >= 13],
        [p for p in primes if p >= 11],
        [p for p in primes if p >= 7],
        [p for p in primes if p >= 5],
        [p for p in primes if p >= 3],
        primes,
        [2] + [p for p in primes if p >= 13],
        [3] + [p for p in primes if p >= 13],
        [5] + [p for p in primes if p >= 13],
        [2, 3] + [p for p in primes if p >= 17],
        [2, 5] + [p for p in primes if p >= 17],
        [3, 5] + [p for p in primes if p >= 17],
    ]
    
    candidates = []
    for bs in base_sets:
        trimmed = smart_trim(bs)
        if trimmed:
            candidates.append(trimmed)
            
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    max_check = 100000 # Task cap
    
    # Evaluate candidates with random restarts
    # We will do 1 deterministic pass, then several random passes if time permits.
    
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline - 10:
            break
        
        # Deterministic Greedy
        classes = greedy_assign(cand_primes, max_check=20000) # Use smaller check for greedy speed
        classes, L = local_search_targeted(classes, max_iter=100, max_check=max_check)
        
        if L > best_L:
            best_L = L
            best_classes = classes
            
        # Random Restarts
        # Try 5 random shuffles of the prime order for greedy
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            random.shuffle(cand_primes)
            classes = greedy_assign(cand_primes, max_check=20000)
            classes, L = local_search_targeted(classes, max_iter=100, max_check=max_check)
            
            if L > best_L:
                best_L = L
                best_classes = classes

    # Final Verification
    if best_classes and verify_classes(best_classes, best_L):
        return best_classes
    
    # Fallback
    return [[2, 1]]
