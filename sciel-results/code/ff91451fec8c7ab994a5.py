"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def _primes_upto(n):
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    primes = _primes_upto(1000)
    # We will consider primes >= 13 to match the seed's spirit, but actually any prime <= 1000 is allowed.
    # The seed used primes from 13 upward. Let's use all primes <= 1000 to maximize flexibility,
    # but the budget constraint is sum(1/p) <= 1.
    
    # Precompute coverage bitmasks for each prime and each residue.
    # To keep memory reasonable and speed high, we limit our search horizon.
    # The seed covers up to ~1000. Let's aim for a window of size W = 2000 or 5000.
    # If we use a window of 5000, a 64-bit int is not enough. We need Python ints or numpy arrays.
    # Python ints are fast for bit operations on small numbers of bits, but 5000 bits is ~600 bytes per int.
    # 168 primes * 1000 residues * 600 bytes = ~100MB. This is okay.
    # However, iterating over 168,000 candidates in Python might be slow.
    # Let's try a smaller window first, say W=1000, and expand if time permits?
    # Actually, the seed scores 1.0 which corresponds to L=999 (or similar).
    # Let's use W = 2000 to try to beat it.
    
    W = 2000
    # We care about covering integers 1..W.
    # Bit 0 corresponds to integer 1. Bit i corresponds to integer i+1.
    
    # Precompute masks for all (p, r) pairs
    # candidates: list of (mask, p, r, cost)
    candidates = []
    for p in primes:
        if p > 1000:
            break
        # Cost is 1/p
        cost = Fraction(1, p)
        for r in range(p):
            # Create mask for integers x in 1..W such that x % p == r
            # x = r, r+p, r+2p, ...
            # But x must be >= 1.
            # If r == 0, x are p, 2p, ...
            # If r > 0, x are r, r+p, ... (if r >= 1)
            
            mask = 0
            # Find first x >= 1
            if r == 0:
                first_x = p
            else:
                first_x = r
                if first_x < 1:
                    first_x = 1 + p * ((1 - r + p - 1) // p) # This logic is messy, just loop
            
            # Efficiently set bits
            # x, x+p, x+2p ... <= W
            # Bit index is x-1
            if first_x <= W:
                for x in range(first_x, W + 1, p):
                    mask |= (1 << (x - 1))
            
            if mask != 0:
                candidates.append((mask, p, r, cost))
    
    # Sort candidates by "value" (number of bits set) descending?
    # Greedy set cover usually picks the one covering most uncovered.
    # We will maintain a `covered` bitmask.
    # To speed up, we can pre-sort by initial density, but the optimal choice depends on `covered`.
    
    covered_mask = 0
    used_primes = set()
    total_cost = Fraction(0)
    classes = []
    
    # We will do a greedy loop.
    # To make it faster, we can limit the number of candidates we evaluate per step.
    # Or we can use a priority queue? But priorities change.
    # Let's just iterate through all candidates each step.
    # 168 primes * avg 500 residues = 84k candidates.
    # 84k candidates * 2000 steps (worst case) = 168M operations. Might be too slow in Python.
    
    # Optimization: Pre-compute popcount for candidates?
    # We need popcount of (cand_mask & ~covered_mask).
    
    # Let's reduce the candidate set.
    # Only consider primes p <= W? Yes, if p > W, each class covers at most 1 number in 1..W.
    # Actually if p > W, r must be in 1..W to cover anything.
    # But small primes cover many numbers.
    
    # Let's filter candidates: only keep those with popcount > 0.
    # And maybe sort by popcount descending initially.
    
    # To speed up the inner loop, we can use numpy for the mask operations?
    # No, Python big ints are fast for bit ops.
    
    # Let's try to limit the search to a smaller set of "promising" candidates.
    # Or just run the greedy loop with a time check.
    
    # Re-organize candidates for faster access
    # List of (mask, p, r, cost)
    
    # We'll iterate until no more candidates can be added (budget) or no improvement.
    
    # To speed up, let's pre-compute the list of candidates once.
    # And in the loop, find the best one.
    
    # A faster approach:
    # 1. Compute all candidate masks.
    # 2. In each step, iterate through candidates.
    #    - If p is already used, skip.
    #    - If total_cost + cost > 1, skip.
    #    - Calculate gain = popcount(mask & ~covered_mask).
    #    - Track max gain.
    # 3. If max gain > 0, add it. Else break.
    
    # To make this fit in time, we might need to restrict the number of primes we consider.
    # The seed used primes from 13. Sum of reciprocals of primes from 13 to 1000 is ~0.92.
    # This leaves little room.
    # Greedy might pick a small prime like 2 or 3 if available, which covers 50% or 33%.
    # But we are restricted to distinct primes.
    
    # Let's run the greedy algorithm.
    
    # Pre-compute popcounts of candidate masks? No, gain depends on covered_mask.
    
    # Let's try to optimize the inner loop by only considering candidates that are not "dominated".
    # This is hard.
    
    # Alternative: Use a smaller W?
    # If W=1000, it's faster. But we want to beat 999.
    # Let's try W=1500.
    
    # Let's refine the candidate generation to be faster.
    
    # Actually, let's just implement the greedy loop and see if it's fast enough.
    # If not, we can reduce W or candidates.
    
    # To ensure we return *something* valid quickly, let's start with the seed solution as a baseline.
    # Seed: primes from 13, r = (i+1) % p.
    # This covers 1..L where L is roughly the number of primes used.
    
    # Let's initialize with the seed solution to guarantee a baseline score of 1.0.
    # Then try to improve.
    
    # Seed Logic:
    # primes = [p for p in range(13, 1001) if is_prime(p)]
    # classes_seed = [[p, (i+1)%p] for i, p in enumerate(primes)]
    
    # Let's calculate the coverage of the seed to get a baseline L.
    
    # But the task asks to maximize L.
    # The seed returns a specific list.
    # My greedy approach might find a better list.
    
    # Let's implement the greedy search with a time limit.
    
    # To speed up candidate generation:
    # We can skip primes p where 1/p is too large? No, small p are good for coverage.
    
    # Let's use W = 2000.
    # Number of candidates: sum(p for p in primes <= 1000) ~ 76k.
    # 76k candidates.
    # Loop: while True:
    #   best_gain = 0
    #   best_cand = None
    #   for cand in candidates:
    #     if p in used: continue
    #     if cost > remaining: continue
    #     gain = popcount(mask & ~covered)
    #     if gain > best_gain:
    #       best_gain = gain
    #       best_cand = cand
    #   if best_gain == 0: break
    #   add best_cand
    
    # This is O(N_candidates * N_steps).
    # N_steps is at most number of primes used ~ 100.
    # 76k * 100 = 7.6M iterations.
    # Each iteration involves big int bit ops and popcount.
    # Python big int popcount is fast (C implementation).
    # This should run in a few seconds.
    
    # Let's code it.
    
    # 1. Generate candidates
    candidates = []
    for p in primes:
        if p > 1000:
            continue
        cost = Fraction(1, p)
        for r in range(p):
            mask = 0
            # x in 1..W, x % p == r
            # first x:
            if r == 0:
                first_x = p
            else:
                first_x = r
                if first_x < 1:
                    first_x = 1 + p * ((1 - r + p - 1) // p)
            
            if first_x <= W:
                # Set bits for first_x, first_x+p, ...
                # Using a loop is slow for large p, but p <= 1000.
                # Max iterations per mask: W/p. Sum over r is W.
                # Total work to generate all masks: sum_p (p * W/p) = sum_p W = 168 * 2000 = 336k.
                # This is very fast.
                for x in range(first_x, W + 1, p):
                    mask |= (1 << (x - 1))
            
            if mask:
                candidates.append((mask, p, r, cost))
    
    # 2. Greedy Selection
    covered_mask = 0
    used_primes = set()
    total_cost = Fraction(0)
    classes = []
    
    # Sort candidates by initial popcount descending to find good ones faster?
    # Not strictly necessary for correctness, but might find good solutions earlier.
    # However, the "best" candidate changes as covered_mask grows.
    
    # We will iterate.
    # To avoid re-scanning all candidates if many are invalid (used or budget),
    # we can just scan all.
    
    while time.time() < deadline:
        best_gain = 0
        best_cand = None
        remaining_budget = Fraction(1) - total_cost
        
        for mask, p, r, cost in candidates:
            if p in used_primes:
                continue
            if cost > remaining_budget:
                continue
            
            # Calculate gain
            # uncovered = ~covered_mask
            # gain_mask = mask & ~covered_mask
            # gain = gain_mask.bit_count()
            
            gain_mask = mask & ~covered_mask
            gain = gain_mask.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_cand = (mask, p, r, cost)
        
        if best_gain == 0:
            break
        
        mask, p, r, cost = best_cand
        covered_mask |= mask
        used_primes.add(p)
        total_cost += cost
        classes.append([p, r])
        
        # Optional: check if we've covered all 1..W
        if covered_mask == (1 << W) - 1:
            break
    
    # Calculate L
    # Find the largest L such that 1..L are covered.
    # This is the position of the first 0 bit in covered_mask, minus 1.
    # If covered_mask has bits 0..L-1 set, then L is the count of trailing 1s?
    # No, it's the first 0 bit index.
    # If covered_mask = 0b1110, first 0 is at index 3 (0-based). So L=3.
    # If covered_mask = 0b1111, first 0 is at index 4. So L=4.
    
    # We can find L by:
    # ~covered_mask & ((1 << W) - 1) gives the uncovered bits within 1..W.
    # If this is 0, then L >= W.
    # Else, the lowest set bit in the uncovered mask is the first gap.
    # gap_pos = (uncovered_mask & -uncovered_mask).bit_length() - 1
    # L = gap_pos
    
    # But what if the gap is beyond W?
    # We capped at W. The score is capped at 10^5.
    # If we cover 1..W, we should check further?
    # The problem says "capped at 10^5".
    # If we cover 1..W, we can try to extend the check.
    # But for now, let's just report the coverage within 1..W.
    # If we cover all 1..W, we can set L = W (or check further if time permits).
    
    # Let's compute L exactly for the range we checked.
    # If we didn't cover all 1..W, L is the first gap.
    # If we did, L is at least W.
    
    # To be safe and maximize score, if we cover 1..W, we can do a quick check for W+1, W+2...
    # But that requires checking if W+k is in any class.
    # (W+k) % p == r for some (p,r) in classes.
    
    # Let's do a quick extension check if time permits.
    
    if covered_mask == (1 << W) - 1:
        # Extend check
        L = W
        # Check up to W + 1000 or until gap
        for k in range(1, 1001):
            n = W + k
            covered = False
            for p, r in classes:
                if n % p == r:
                    covered = True
                    break
            if covered:
                L = n
            else:
                break
    else:
        # Find first gap
        uncovered = (~covered_mask) & ((1 << W) - 1)
        if uncovered == 0:
            L = W
        else:
            # Lowest set bit
            lowest_bit = uncovered & -uncovered
            L = lowest_bit.bit_length() - 1
    
    # Ensure we return a valid list
    # The list `classes` contains [p, r] pairs.
    # We must ensure sum(1/p) <= 1.
    # Our greedy algorithm ensures this.
    
    return classes

# EVOLVE-BLOCK-END
