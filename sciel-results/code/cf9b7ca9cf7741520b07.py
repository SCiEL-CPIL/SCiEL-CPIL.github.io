"""
Strategy: 
1. Determine the LCM of the divisors to understand the periodicity of the problem.
2. We want to find 10 distinct integers x_i such that x_i % d_i == 0.
3. Let the interval be [L, L+W]. We want to minimize W.
4. Instead of backtracking on values directly, we can model this as an Exact Cover problem (DLX) 
   or simply a bipartite matching problem if we fix the set of available slots.
   However, the "slots" are all integers in [L, L+W].
   For a fixed W, we can check if a valid assignment exists.
   Since W is likely small (relative to the divisors), we can iterate W from 0 upwards? 
   No, W can be up to ~1000.
   
   Better approach:
   The optimal solution will have a span W. 
   The values x_i are multiples of d_i.
   Let's iterate over possible starting positions L in a smart way.
   The LCM of [6, 10, 14, 15, 21, 22, 26, 33, 35, 39] is 1051050? No, let's calculate.
   6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13.
   Primes involved: 2, 3, 5, 7, 11, 13.
   LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030.
   Wait, 22 has 11, 33 has 11. 26 has 13, 39 has 13.
   So LCM is 30030.
   
   Any valid solution [x_1, ..., x_10] can be shifted by k*30030 to another valid solution.
   Therefore, we only need to search for the optimal window [L, L+W] where L is in [0, 30030-1] 
   relative to the "phase" of the multiples? 
   Actually, the set of residues mod 30030 for each divisor is fixed.
   Let R_i be the set of residues r in [0, 30029] such that r % d_i == 0.
   We want to pick one residue r_i for each i, and an integer offset k_i such that 
   x_i = r_i + 30030 * k_i.
   And all x_i must be distinct and within [L, L+W].
   
   This is still complex. Let's use a simpler heuristic search:
   1. The span W is likely small. Let's estimate the density.
      Each divisor d_i covers 1/d_i of the integers.
      With 10 divisors, the "conflict" is high.
      The expected gap between consecutive valid numbers for a single divisor is d_i.
      For 10 divisors, we are picking 10 numbers.
      A rough lower bound on W is related to the largest divisor? 
      Actually, if we pick multiples of the LCM, they are all 30030 apart. That's huge.
      We want them close.
      
   Let's try a sliding window approach with a feasibility check.
   For a given window size W, does there exist an L such that we can pick distinct multiples?
   We can check this using Maximum Bipartite Matching.
   Left side: Divisors d_1...d_10.
   Right side: Integers in [L, L+W].
   Edge exists if integer is divisible by divisor.
   We need a perfect matching.
   
   How many L's to check?
   The pattern of divisibility repeats every LCM=30030.
   So we only need to check L in [0, 30029].
   For each L, we check if a perfect matching exists for window [L, L+W].
   We can binary search on W? Or just iterate W from 0 upwards?
   W is likely < 1000.
   Checking a matching of size 10xW is fast.
   But 30030 * 1000 * matching_cost might be too slow.
   
   Optimization:
   We don't need to check every L.
   If a solution exists with span W, the "critical" positions are determined by the residues.
   
   Alternative: 
   Randomized Local Search / Simulated Annealing.
   Start with a random valid assignment (e.g., x_i = d_i * random_int).
   Calculate span.
   Try to move one x_i to a nearby multiple of d_i to reduce span while maintaining distinctness.
   
   Given the small size (10 items), DLX or backtracking with pruning is very fast.
   Let's combine:
   1. Iterate L from 0 to LCM-1.
   2. For each L, try to find the minimal W using a greedy/backtracking approach that builds the set of numbers in increasing order.
   
   Actually, the "backtrack" in the observed child (p5) scored 2.0625 (value ~ 97? 1000/2.06 = 485? No, score = seed/value. Seed value was likely 1000? No, seed program returns divisors. 
   Seed program: return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
   Max = 39, Min = 6. Span = 33.
   Score = 1.0. So Seed Value = 33.
   Child p5 score 2.0625 => Value = 33 / 2.0625 = 16.
   So the current best value is 16.
   
   Can we beat 16?
   Divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
   If span is 16, the numbers fit in [L, L+16].
   There are 17 integers in the window. We need to pick 10.
   Each number must be divisible by its specific divisor.
   
   Let's write a fast solver.
   We will iterate L from 0 to LCM-1 (30030).
   For each L, we will try to find the minimal W.
   We can do this by expanding W from 0 upwards? 
   Or, for a fixed W, check if matching exists.
   Since W is small (<= 16), the graph is tiny.
   Matching 10 nodes to <= 17 nodes is trivial.
   
   Algorithm:
   1. Precompute LCM = 30030.
   2. best_val = 33 (from seed).
   3. For W in range(0, best_val):  # We want to find if any W < best_val works.
        For L in range(0, LCM):
            Check if there is a perfect matching for window [L, L+W].
            If yes, update best_val = W and break out of L loop? 
            No, we should find the MIN W. So we iterate W from 0 upwards.
            As soon as we find ANY valid L for a given W, that W is the minimum possible?
            Yes, because we are iterating W in increasing order.
            
   Complexity:
   Max W to check: 16 (current best).
   L range: 30030.
   Total checks: 17 * 30030 ~ 500,000.
   Each check: Bipartite matching on 10 x 17 graph.
   Hopcroft-Karp or even DFS-based matching (Kuhn's algorithm) is O(E*V) ~ 170 * 10 = 1700 ops.
   500,000 * 1700 = 850,000,000 ops. 
   This might be too slow in Python (1-2 seconds per 100M ops? No, Python is slower, ~10M-50M ops/sec).
   850M ops could take 20-80 seconds. Risky.
   
   Optimization:
   We don't need to check ALL L for each W.
   The condition "x_i is divisible by d_i" depends on L mod d_i.
   The pattern of available candidates for a divisor d_i in window [L, L+W] changes as L changes.
   However, we can prune.
   
   Better Optimization:
   Use the fact that if a solution exists for L, it might exist for L+1? No.
   
   Let's try to reduce the L range.
   The "phase" of the window relative to the divisors is determined by L mod LCM.
   But we can be smarter.
   For a fixed W, the set of valid (L, assignment) pairs is periodic with period LCM.
   
   Can we use a heuristic to skip L?
   Or, can we solve for L given W?
   
   Alternative:
   Random Search + Local Search.
   1. Generate random L in [0, LCM).
   2. For this L, find the minimal W using a greedy backtracking that tries to pack numbers tightly.
   3. Keep track of best W.
   4. Run for a few seconds.
   
   Given the small search space for W (0..16), and the periodicity, 
   maybe we can just check a subset of L?
   
   Actually, let's look at the constraints.
   d_i are: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
   If W=16, window size 17.
   The number of multiples of d in [L, L+16] is either floor(16/d) or ceil(16/d).
   For d=6: 2 or 3 multiples.
   For d=10: 1 or 2.
   For d=14: 1 or 2.
   For d=15: 1 or 2.
   For d=21: 1.
   For d=22: 1.
   For d=26: 1.
   For d=33: 1.
   For d=35: 1.
   For d=39: 1.
   
   Most divisors have only 1 candidate in a window of size 16!
   Specifically, if d > 16, there is at most 1 multiple in any window of size 16.
   Divisors > 16: 21, 22, 26, 33, 35, 39. (6 divisors)
   These MUST map to their unique multiple in the window.
   So for a given L, the values for these 6 divisors are FIXED (if they exist in the window).
   If any of these 6 don't have a multiple in [L, L+W], the window is invalid for that L.
   If they do, they are fixed.
   Then we have 4 divisors left: 6, 10, 14, 15.
   We need to assign them to the remaining slots in [L, L+W] such that they are distinct from the fixed ones and from each other.
   
   This drastically simplifies the problem!
   
   Algorithm:
   1. LCM = 30030.
   2. best_W = 33.
   3. For W in range(0, best_W):
        For L in range(0, LCM):
            # Check if the "large" divisors have exactly one multiple in [L, L+W]
            # If yes, those values are forced.
            # If no (0 or >1), it's still possible, but for W < 21, d>16 implies 0 or 1.
            # Actually for W=16, d=21 -> 0 or 1.
            
            forced_vals = []
            valid = True
            # Divisors > W are "rigid" (0 or 1 candidates)
            # Divisors <= W are "flexible"
            
            # Let's just use a generic check for all divisors, but optimize the "rigid" ones.
            
            # For each divisor d_i:
            # Find multiples in [L, L+W].
            # If count == 0: Invalid L for this W.
            # If count == 1: Value is forced.
            # If count > 1: Value is flexible.
            
            # If any forced values collide (duplicate), Invalid.
            # If forced values are valid, check if we can assign the flexible ones
            # to the remaining slots without collision.
            
            # Since flexible divisors are small (6, 10, 14, 15), and W is small,
            # the number of flexible candidates is small.
            # We can use a simple backtracking or matching for the flexible part.
            
            # This check is very fast.
            
            if check_window(L, W):
                best_W = W
                break # Found min W for this W? No, we are iterating W. If found, this W is the global min.
        if best_W == W:
            break # Found the absolute minimum W.
            
   Return the solution corresponding to best_W.
   
   Time complexity:
   W iterates 0..16.
   L iterates 0..30029.
   Total 17 * 30030 ~ 500k checks.
   Each check:
   - Find multiples for 10 divisors. 
     For d > W, we can calculate the unique multiple directly:
     m = ceil(L/d)*d. If m <= L+W, it's the only one.
   - Check collisions.
   - Match flexible ones.
   
   This should be fast enough in Python if optimized.
   500k * (small constant) < 1 second.
   
   We need to reconstruct the solution for the best W and L.
   
   Let's implement this.
"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    best_W = 33  # Seed span
    best_L = 0
    best_solution = None
    
    # We iterate W from 0 upwards to find the minimum span.
    # We stop as soon as we find a valid L for a given W.
    for W in range(0, best_W + 1):
        # To avoid timeout, we can limit L checks if W is large, 
        # but W is expected to be small.
        for L in range(0, LCM):
            if time.time() > deadline:
                break
                
            # Check if window [L, L+W] allows a valid assignment
            sol = find_assignment(divisors, L, W)
            if sol is not None:
                best_W = W
                best_L = L
                best_solution = sol
                break # Found min W for this W, so this is the global min W
        if best_solution is not None:
            break
            
    if best_solution is None:
        # Fallback to seed
        return P['divisors'][:]
        
    return best_solution

def find_assignment(divisors, L, W):
    """
    Returns a list of 10 distinct integers in [L, L+W] divisible by corresponding divisors,
    or None if not possible.
    """
    n = len(divisors)
    used_values = set()
    assignments = [None] * n
    flexible_indices = []
    
    # 1. Handle rigid divisors (d > W)
    # For d > W, there is at most one multiple in [L, L+W].
    for i, d in enumerate(divisors):
        if d > W:
            # Find the multiple of d in [L, L+W]
            # k = ceil(L / d)
            k = (L + d - 1) // d
            val = k * d
            if val <= L + W:
                # Check if val is already used
                if val in used_values:
                    return None
                used_values.add(val)
                assignments[i] = val
            else:
                # No multiple in range
                return None
        else:
            flexible_indices.append(i)
            
    # 2. Handle flexible divisors
    # We need to assign values to flexible_indices such that:
    # - val is multiple of d
    # - val is in [L, L+W]
    # - val is not in used_values
    # - all val are distinct
    
    # Generate candidates for each flexible index
    candidates = []
    for i in flexible_indices:
        d = divisors[i]
        cands = []
        k = (L + d - 1) // d
        val = k * d
        while val <= L + W:
            if val not in used_values:
                cands.append(val)
            val += d
        candidates.append(cands)
        
    # Backtracking to assign flexible indices
    # Order flexible indices by number of candidates (MRV heuristic)
    # indices in flexible_indices correspond to candidates list
    order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
    
    def backtrack(idx):
        if idx == len(order):
            return True
        flex_idx_in_list = order[idx]
        original_idx = flexible_indices[flex_idx_in_list]
        for val in candidates[flex_idx_in_list]:
            if val not in used_values:
                used_values.add(val)
                assignments[original_idx] = val
                if backtrack(idx + 1):
                    return True
                used_values.remove(val)
                assignments[original_idx] = None
        return False
        
    if backtrack(0):
        return assignments
    else:
        return None

# EVOLVE-BLOCK-END
