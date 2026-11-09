import time
import random
from fractions import Fraction
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    prime_set = set(primes)
    MAX_L = 1000
    
    # Precompute masks for all primes and residues
    # mask[p][r] is a bitset of integers n in 1..1000 such that n % p == r
    masks = {}
    for p in primes:
        masks[p] = [0] * p
        for r in range(p):
            mask = 0
            if r == 0:
                start = p
            else:
                start = r
            n = start
            while n <= MAX_L:
                mask |= (1 << (n - 1))
                n += p
            masks[p][r] = mask
            
    full_mask = (1 << MAX_L) - 1
    
    # Helper to calculate L from a cover bitset
    def calc_L(cover):
        if (cover & full_mask) == full_mask:
            return MAX_L
        uncovered = (~cover) & full_mask
        if uncovered == 0:
            return MAX_L
        low_bit = uncovered & -uncovered
        idx = low_bit.bit_length() - 1
        return idx

    # Helper to calculate total cost
    def calc_cost(selected_primes):
        c = Fraction(0)
        for p in selected_primes:
            c += Fraction(1, p)
        return c

    # Helper to evaluate a solution: returns (L, cost, is_valid)
    def evaluate_solution(classes):
        if not classes:
            return 0, Fraction(0), True
        
        seen = set()
        for p, r in classes:
            if p in seen:
                return 0, Fraction(1), False
            seen.add(p)
            if r < 0 or r >= p:
                return 0, Fraction(1), False
            if p > 1000:
                return 0, Fraction(1), False
                
        cover = 0
        for p, r in classes:
            cover |= masks[p][r]
            
        L = calc_L(cover)
        cost = calc_cost(seen)
        
        if cost > 1:
            return 0, cost, False
            
        return L, cost, True

    # --- Phase 1: Initial Heuristic ---
    # Start with a greedy set of small primes which are "efficient" in terms of coverage density
    # but expensive in budget. We will optimize the set later.
    
    # A good starting set: Primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Sum 1/p for first 25 primes is roughly 1.0?
    # Let's check sum of 1/p for primes up to 100 is ~1.0.
    # Actually sum 1/p diverges slowly.
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59+1/61+1/67+1/71+1/73+1/79+1/83+1/89+1/97
    # This sum is approximately 1.0.
    
    initial_primes = [p for p in primes if p <= 97]
    # Check cost
    cost_init = sum(Fraction(1, p) for p in initial_primes)
    if cost_init > 1:
         # Remove largest primes until cost <= 1
         while cost_init > 1 and initial_primes:
             # Remove largest prime to keep small ones
             p_to_remove = max(initial_primes)
             initial_primes.remove(p_to_remove)
             cost_init -= Fraction(1, p_to_remove)
             
    # For each prime, find the best residue to cover 1..L
    # Greedy: pick r that maximizes coverage of 1..L
    current_classes = []
    current_cover = 0
    for p in initial_primes:
        best_r = 0
        best_gain = -1
        for r in range(p):
            m = masks[p][r]
            new_bits = m & ~current_cover
            gain = new_bits.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
        current_classes.append((p, best_r))
        current_cover |= masks[p][best_r]
        
    best_L, best_cost, is_valid = evaluate_solution(current_classes)
    best_solution = [list(x) for x in current_classes]
    
    # --- Phase 2: Local Search on Prime Set and Residues ---
    # We use a simulated annealing approach to modify the set of primes and their residues.
    # Moves:
    # 1. Swap a prime in the set with a prime not in the set.
    # 2. Change the residue of a prime in the set.
    # 3. Add a prime (if budget allows).
    # 4. Remove a prime.
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    random.seed(42)
    
    current_L = best_L
    current_classes_list = [list(x) for x in best_solution]
    current_cover = 0
    for p, r in current_classes_list:
        current_cover |= masks[p][r]
    current_cost = calc_cost([p for p, r in current_classes_list])
    current_primes_set = set(p for p, r in current_classes_list)
    
    # Precompute list of all primes for random selection
    all_primes = primes
    
    # To speed up evaluation, we can try to use the Hungarian algorithm idea?
    # No, the task is to cover 1..L, not assign specific integers to primes.
    # The "inspiration" suggested max flow/matching.
    # A direct matching approach:
    # We want to cover 1..L.
    # If we fix the set of primes, the optimal residues can be found by checking which residues cover the most "uncovered" integers.
    # However, residues interact.
    # Let's stick to SA but make it faster by incremental updates where possible,
    # or by using a smarter move: "Optimize residues for current set" using a greedy pass.
    
    def optimize_residues(prime_list, current_cover_base=0):
        """
        Given a set of primes, find residues that maximize coverage of 1..MAX_L.
        This is still NP-hard in general, but a greedy pass is fast.
        """
        cover = current_cover_base
        res = {}
        # Sort primes by size? Smaller primes have fewer choices, might be easier to place first?
        # Or larger primes have more choices.
        # Let's just iterate.
        for p in prime_list:
            best_r = 0
            best_gain = -1
            for r in range(p):
                m = masks[p][r]
                new_bits = m & ~cover
                gain = new_bits.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            res[p] = best_r
            cover |= masks[p][best_r]
        return res, cover

    # In the SA loop, we will perform "Residue Optimization" moves frequently.
    # A move: Pick a random subset of primes (e.g., 2-5) and re-optimize their residues greedily given the others.
    
    while time.time() < deadline and T > min_T:
        move_type = random.randint(0, 3)
        new_classes = [list(x) for x in current_classes_list]
        new_primes_set = set(p for p, r in new_classes)
        
        if move_type == 0: # Swap Prime
            if new_classes:
                idx = random.randint(0, len(new_classes) - 1)
                old_p, old_r = new_classes[idx]
                # Pick a random prime not in set
                candidates = [p for p in all_primes if p not in new_primes_set]
                if candidates:
                    new_p = random.choice(candidates)
                    # Check budget
                    old_cost = Fraction(1, old_p)
                    new_cost = Fraction(1, new_p)
                    if current_cost - old_cost + new_cost <= 1:
                        new_classes[idx] = [new_p, old_r if old_r < new_p else 0] # Initialize r
                        new_primes_set.discard(old_p)
                        new_primes_set.add(new_p)
                        # Re-optimize residues for the whole set? Or just the changed one?
                        # Re-optimizing the whole set is expensive (O(N * P * L/word)).
                        # Let's re-optimize residues for the whole set in this move.
                        new_res, new_cover = optimize_residues([p for p, _ in new_classes])
                        new_classes = [[p, new_res[p]] for p in new_res]
                        new_cost_total = calc_cost(new_primes_set)
                        
                        new_L = calc_L(new_cover)
                        if new_L > current_L or (random.random() < 2.71828 ** ((new_L - current_L) / T)):
                            current_L = new_L
                            current_classes_list = new_classes
                            current_cover = new_cover
                            current_cost = new_cost_total
                            current_primes_set = new_primes_set
                            if current_L > best_L:
                                best_L = current_L
                                best_solution = [list(x) for x in current_classes_list]
                        T *= cooling
                        continue # Skip standard acceptance logic below
                    else:
                        T *= cooling
                        continue
            else:
                T *= cooling
                continue
                
        elif move_type == 1: # Add Prime
            affordable = [p for p in all_primes if p not in new_primes_set and current_cost + Fraction(1, p) <= 1]
            if affordable:
                p = random.choice(affordable)
                # Find best r for this p given current cover
                best_r = 0
                best_gain = -1
                for r in range(p):
                    m = masks[p][r]
                    new_bits = m & ~current_cover
                    gain = new_bits.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                new_classes.append([p, best_r])
                new_primes_set.add(p)
                new_cover = current_cover | masks[p][best_r]
                new_cost_total = current_cost + Fraction(1, p)
                
                new_L = calc_L(new_cover)
                if new_L > current_L or (random.random() < 2.71828 ** ((new_L - current_L) / T)):
                    current_L = new_L
                    current_classes_list = new_classes
                    current_cover = new_cover
                    current_cost = new_cost_total
                    current_primes_set = new_primes_set
                    if current_L > best_L:
                        best_L = current_L
                        best_solution = [list(x) for x in current_classes_list]
                T *= cooling
                continue
            else:
                T *= cooling
                continue

        elif move_type == 2: # Remove Prime
            if new_classes:
                idx = random.randint(0, len(new_classes) - 1)
                p, r = new_classes.pop(idx)
                new_primes_set.discard(p)
                # Recalculate cover
                new_cover = 0
                for pp, rr in new_classes:
                    new_cover |= masks[pp][rr]
                new_cost_total = current_cost - Fraction(1, p)
                
                new_L = calc_L(new_cover)
                if new_L > current_L or (random.random() < 2.71828 ** ((new_L - current_L) / T)):
                    current_L = new_L
                    current_classes_list = new_classes
                    current_cover = new_cover
                    current_cost = new_cost_total
                    current_primes_set = new_primes_set
                    if current_L > best_L:
                        best_L = current_L
                        best_solution = [list(x) for x in current_classes_list]
                T *= cooling
                continue
            else:
                T *= cooling
                continue
                
        elif move_type == 3: # Re-optimize Residues for a random subset
            if new_classes:
                # Pick 2-4 random primes to re-optimize
                k = random.randint(2, min(4, len(new_classes)))
                indices = random.sample(range(len(new_classes)), k)
                subset_primes = [new_classes[i][0] for i in indices]
                
                # Calculate cover of the rest
                rest_cover = 0
                for i, (p, r) in enumerate(new_classes):
                    if i not in indices:
                        rest_cover |= masks[p][r]
                        
                # Greedily optimize residues for subset_primes given rest_cover
                # Order matters. Sort by p descending? 
                # Let's try sorting by p ascending (smaller primes have fewer choices, might be more constrained)
                # Actually, larger primes have more "flexibility" in residues.
                # Let's just use the order they appear.
                new_res = {}
                temp_cover = rest_cover
                # We need to assign residues to subset_primes.
                # Simple greedy: pick p, find best r, update cover, next p.
                # To make it better, we can try a few random orders.
                # For speed, just one order.
                for p in subset_primes:
                    best_r = 0
                    best_gain = -1
                    for r in range(p):
                        m = masks[p][r]
                        new_bits = m & ~temp_cover
                        gain = new_bits.bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    new_res[p] = best_r
                    temp_cover |= masks[p][best_r]
                    
                # Apply new residues
                for i in indices:
                    p = new_classes[i][0]
                    new_classes[i][1] = new_res[p]
                    
                new_cover = temp_cover
                # Cost unchanged
                new_L = calc_L(new_cover)
                
                if new_L > current_L or (random.random() < 2.71828 ** ((new_L - current_L) / T)):
                    current_L = new_L
                    current_classes_list = new_classes
                    current_cover = new_cover
                    # current_cost unchanged
                    if current_L > best_L:
                        best_L = current_L
                        best_solution = [list(x) for x in current_classes_list]
                T *= cooling
                continue

    # Final Verification
    final_L, final_cost, valid = evaluate_solution(best_solution)
    if not valid:
        best_solution = [[2, 1]]
        final_L, final_cost, valid = evaluate_solution(best_solution)
        
    return best_solution

# EVOLVE-BLOCK-END
