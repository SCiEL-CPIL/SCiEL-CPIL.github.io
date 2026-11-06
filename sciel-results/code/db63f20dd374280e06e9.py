import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin for 150s limit

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
    # We want to maximize coverage.
    # The sum of 1/p must be <= 1.
    # Sum of reciprocals of primes 13..997 is approx 0.92.
    # We can potentially add smaller primes if we remove larger ones, or just use the set that maximizes "coverage power".
    # Smaller primes have higher "density" (1/p is larger) but also cover more frequently (period p is smaller).
    # However, 1/p is the cost.
    # Let's stick to primes >= 13 as a baseline, but maybe try to optimize the set if time permits.
    # Actually, including 11 or 7 might be beneficial if we drop some large primes.
    # Let's try a few sets.
    
    # Set A: Primes 13..997
    set_A = [p for p in all_primes if p >= 13]
    sum_A = sum(1/p for p in set_A)
    
    # Set B: Primes 11..997, drop some large ones to keep sum <= 1?
    # Sum of 1/11 + 1/13 + ... is roughly 0.92 + 0.09 = 1.01 > 1.
    # So we can't just add 11. We'd have to drop some large primes.
    # Let's stick to Set A for simplicity and proven performance, but maybe try adding 11 and removing the largest few.
    
    # Let's try to construct a set that includes 11.
    # Sum of 1/p for p in [11, 997] is ~1.01.
    # We need to reduce by ~0.01.
    # Largest primes have smallest 1/p. Removing them saves little.
    # We need to remove primes with larger 1/p? No, that increases sum.
    # We need to remove primes to *decrease* the sum.
    # So we remove primes. To keep the set "strong", we should keep small primes.
    # So remove largest primes.
    
    # Let's try Set B: Start with 11..997, remove largest primes until sum <= 1.
    candidates = [p for p in all_primes if p >= 11]
    sum_c = sum(1/p for p in candidates)
    # Sort by value descending (largest first)
    candidates_sorted = sorted(candidates, reverse=True)
    primes_set = []
    current_sum = 0
    # We want to keep as many as possible, prioritizing smaller primes?
    # Actually, for covering, smaller primes are generally better because they cover more numbers in a range?
    # No, the density is 1/p.
    # But smaller primes have smaller periods, so they "interfere" less? Or more?
    # Let's just try the set that includes 11 and 13 and drops the largest ones to fit budget.
    
    # Greedy: Keep smallest primes, add larger ones if budget allows.
    # This maximizes the number of primes? No, sum constraint.
    # To maximize coverage, we want a mix.
    
    # Let's just use Set A (13..997) as it's a known good baseline.
    # But let's check if we can improve by swapping.
    # For now, use Set A.
    primes_set = set_A
    
    # Verify sum with exact rationals to be safe, though float is likely fine for selection.
    # The problem statement says "exact rationals" for the constraint check.
    # We must ensure sum(1/p) <= 1.
    exact_sum = sum(Fraction(1, p) for p in primes_set)
    if exact_sum > 1:
        # Fallback: remove largest primes until sum <= 1
        primes_set = sorted(primes_set, reverse=True)
        while exact_sum > 1 and primes_set:
            p = primes_set.pop()
            exact_sum -= Fraction(1, p)
        primes_set = sorted(primes_set) # Restore order for indexing? Or just keep as is.
    
    n_primes = len(primes_set)
    L_MAX = 100000
    # We will check coverage up to a limit. If we reach the limit, we assume we can go further?
    # No, we must find the EXACT L.
    # But calculating up to 100,000 for every step is expensive.
    # Strategy: 
    # 1. Optimize residues to cover as much as possible.
    # 2. At the end, calculate the exact L up to 100,000.
    # During search, we can estimate L or check up to a smaller limit (e.g., 5000) to guide search.
    
    L_CHECK_LIMIT = 5000 # Small limit for fast evaluation during search
    L_FINAL_LIMIT = 100000
    
    def evaluate_coverage_exact(residues, L_limit):
        """
        Returns the largest L such that 1..L is covered.
        """
        # Use a bytearray for cover count? Or just check if covered.
        # Since we just need to know if covered (count > 0), we can use a boolean array or bitset.
        # But we need to update it.
        # Let's use a list of counts for incremental updates.
        cover_count = [0] * (L_limit + 1)
        for i, p in enumerate(primes_set):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            elif start > L_limit:
                continue
            for n in range(start, L_limit + 1, p):
                cover_count[n] += 1
        
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L, cover_count

    def calculate_L_from_counts(cover_count, L_limit):
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L

    def get_holes(cover_count, L_limit):
        holes = []
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                holes.append(n)
        return holes

    def update_counts(cover_count, p, r_old, r_new, L_limit):
        if r_old == r_new:
            return
        # Decrement
        start = r_old
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] -= 1
        
        # Increment
        start = r_new
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] += 1

    # Initialization
    def init_residues(seed_style=True):
        res = [0] * n_primes
        if seed_style:
            # Heuristic: Cover i+1 with i-th prime
            for i, p in enumerate(primes_set):
                target = i + 1
                res[i] = target % p
        else:
            for i in range(n_primes):
                res[i] = random.randint(0, primes_set[i] - 1)
        return res

    best_L = 0
    best_classes = None

    # Main Search Loop
    # We will run local search on the small limit, then verify on large limit at the end.
    # To maximize the chance of finding a large L, we should focus on extending the prefix.
    
    # Start with seed-style init
    residues = init_residues(seed_style=True)
    L, cover_count = evaluate_coverage_exact(residues, L_CHECK_LIMIT)
    
    if L > best_L:
        best_L = L
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]

    # Local Search
    # We want to extend L.
    # If L < L_CHECK_LIMIT, there is a hole at L+1.
    # Try to fix the hole.
    
    stuck_counter = 0
    max_stuck = 2000
    
    while time.time() - start_time < DEADLINE:
        if L >= L_CHECK_LIMIT:
            # We've covered the check limit.
            # We can try to extend the check limit?
            # Or just break and do final evaluation.
            # Let's try to extend the limit slightly to see if we can go further.
            # But updating cover_count to a larger limit is expensive (re-init).
            # Let's just break and do the final exact check on the best candidate found so far.
            # But maybe we can do a few more steps with a larger limit?
            # For now, break.
            break
        
        # Find the first hole
        hole = L + 1
        
        # Try to cover this hole by changing a residue.
        # Which prime to change?
        # Random? Or the one that "should" cover it?
        # Let's try all primes and see which change improves L the most (or just if it covers the hole).
        # But changing one prime might uncover previous numbers.
        # We need to ensure we don't reduce L.
        
        improved = False
        # Try random primes to avoid getting stuck in local minima
        # Or try all primes? n_primes is ~140. 140 updates per step is fast.
        
        best_new_L = L
        best_idx = -1
        best_r_new = -1
        
        for idx in range(n_primes):
            p = primes_set[idx]
            r_old = residues[idx]
            r_new = hole % p
            
            if r_old == r_new:
                continue
                
            # Simulate change
            # We need to check if this change covers the hole and doesn't uncover 1..L.
            # Since we are changing one prime, the only numbers affected are those congruent to r_old or r_new mod p.
            # We need to check if any n in 1..L becomes uncovered.
            # This is expensive to check exactly for every candidate.
            # Approximation: Just apply, check L, revert if worse.
            
            update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT)
            residues[idx] = r_new
            
            new_L = calculate_L_from_counts(cover_count, L_CHECK_LIMIT)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_idx = idx
                best_r_new = r_new
                # Don't revert yet, keep this state as candidate?
                # But we want to test other candidates.
                # We should revert this one to test the next one fairly.
                # Wait, if we keep it, we can't test others against the original state.
                # So revert.
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old
            else:
                # Revert
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old
        
        if best_idx != -1:
            # Apply the best change found
            p = primes_set[best_idx]
            r_old = residues[best_idx]
            update_counts(cover_count, p, r_old, best_r_new, L_CHECK_LIMIT)
            residues[best_idx] = best_r_new
            L = best_new_L
            improved = True
            stuck_counter = 0
            
            if L > best_L:
                best_L = L
                best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
        else:
            # No improvement found
            stuck_counter += 1
            # If stuck, try a random restart or random move
            if stuck_counter > max_stuck:
                # Random restart
                residues = init_residues(seed_style=False)
                L, cover_count = evaluate_coverage_exact(residues, L_CHECK_LIMIT)
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                stuck_counter = 0
                max_stuck = 2000 # Reset threshold
    
    # Final Evaluation
    # Take the best_classes found and calculate the exact L up to L_FINAL_LIMIT.
    if best_classes is None:
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    # Reconstruct residues from best_classes
    # best_classes is a list of [p, r]
    # We need to map back to the order of primes_set?
    # The evaluator just checks the returned list.
    # But we need to calculate L for the returned list.
    
    # Let's calculate L for the best_classes list directly.
    # We need to be careful: best_classes might not be in the same order as primes_set.
    # But the logic is the same: sum of 1/p <= 1 (already checked for the set), and coverage.
    
    # Create a map of prime to residue for the best solution
    res_map = {}
    for p, r in best_classes:
        res_map[p] = r
    
    # Calculate coverage up to L_FINAL_LIMIT
    # This might take a while. 100,000 * (sum 1/p) ~ 100,000 * 0.92 ~ 92,000 operations.
    # Actually, for each prime p, we iterate L/p times.
    # Sum(L/p) = L * sum(1/p) ~ 92,000.
    # This is very fast.
    
    cover_count_final = [0] * (L_FINAL_LIMIT + 1)
    for p, r in best_classes:
        start = r
        if start == 0:
            start = p
        elif start > L_FINAL_LIMIT:
            continue
        for n in range(start, L_FINAL_LIMIT + 1, p):
            cover_count_final[n] += 1
            
    final_L = 0
    for n in range(1, L_FINAL_LIMIT + 1):
        if cover_count_final[n] == 0:
            break
        final_L = n
    
    return best_classes

# EVOLVE-BLOCK-END
