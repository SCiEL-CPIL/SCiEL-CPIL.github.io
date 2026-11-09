"""Optimized search for distinct multiples of divisor set using bitset matching."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    # LCM of all divisors
    lcm_all = reduce(_lcm, divisors)
    
    best_val = float('inf')
    best_assign = None

    # For a fixed window [L, L+W], we want to assign each divisor i to a distinct position p_i in [L, L+W]
    # such that p_i is a multiple of divisors[i].
    # Let pos_j = L + j for j in 0..W.
    # Divisor i can be assigned to position j if (L+j) % divisors[i] == 0.
    
    # We can model this as a bipartite matching problem.
    # Left side: divisors (0..n-1)
    # Right side: positions in window (0..W)
    # Edge exists if (L+j) % divisors[i] == 0.
    
    # Since W is small, we can use a bitset for each divisor representing which positions are valid.
    # Then we can use a simple backtracking with bit operations or a greedy matching with backtracking.
    # Given the small size (n=10, W < 100), a simple DFS matching is sufficient and fast.
    
    def find_matching(W, L):
        """
        Try to find a perfect matching for divisors to positions in [L, L+W].
        Returns a list of assigned values (multiples) or None if no matching exists.
        """
        # candidate_mask[i] is a bitset where bit j is 1 if (L+j) is a multiple of divisors[i]
        candidate_mask = [0] * n
        
        for i, d in enumerate(divisors):
            # We need (L + j) % d == 0 => j % d == (-L) % d
            # Let r = (-L) % d. Then j must be a multiple of d plus r? No.
            # (L+j) = k*d => j = k*d - L.
            # So j is in the set { k*d - L | k is integer } intersect [0, W].
            # Let j_0 = (d - (L % d)) % d. This is the smallest non-negative j such that (L+j) % d == 0.
            # Then valid j are j_0, j_0 + d, j_0 + 2d, ... <= W.
            
            j_0 = (d - (L % d)) % d
            if j_0 > W:
                return None  # No multiple in window for this divisor
            
            mask = 0
            for j in range(j_0, W + 1, d):
                mask |= (1 << j)
            
            candidate_mask[i] = mask
        
        # Now we have for each divisor i, a bitmask of valid positions.
        # We need to assign distinct positions to each divisor.
        # This is a bipartite matching problem.
        # We can use a simple DFS with memoization or just backtracking with bitsets.
        
        # Let's use a simple recursive backtracking with bitsets for speed.
        # Order divisors by number of available positions (MRV) to prune early.
        # However, the order of divisors is fixed for the output, but for the search,
        # we can process them in any order and map back.
        
        # Let's create an order of divisors based on popcount of candidate_mask
        order = sorted(range(n), key=lambda i: bin(candidate_mask[i]).count('1'))
        
        # perm[i] will store the position index (0..W) assigned to divisor order[i]
        perm = [None] * n
        used_mask = 0  # Bitset of used positions
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            d_idx = order[idx]
            avail = candidate_mask[d_idx] & ~used_mask
            
            # Try each available position
            # We can iterate through bits in avail
            a = avail
            while a:
                # Get lowest set bit
                lsb = a & -a
                pos_idx = lsb.bit_length() - 1
                
                # Assign this position to divisor d_idx
                perm[d_idx] = pos_idx
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                perm[d_idx] = None
                a ^= lsb
            
            return False
        
        if backtrack(0):
            # Reconstruct the actual values
            result = [0] * n
            for i in range(n):
                result[i] = L + perm[i]
            return result
        else:
            return None

    # Search W from 9 upwards
    # Theoretical minimum W is 9 (10 distinct integers).
    # However, due to gaps in multiples, W will likely be larger.
    # We can start from a slightly higher bound if we want, but 9 is safe.
    
    for W in range(9, 100):
        if time.time() > deadline:
            break
        
        # Iterate L from 0 to lcm_all - 1
        # Due to periodicity, checking L in [0, lcm_all) is sufficient.
        for L in range(lcm_all):
            if time.time() > deadline:
                break
            
            # Quick check: does each divisor have at least one multiple in [L, L+W]?
            # This is implicitly checked in find_matching, but we can pre-filter.
            feasible = True
            for d in divisors:
                j_0 = (d - (L % d)) % d
                if j_0 > W:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            result = find_matching(W, L)
            if result is not None:
                # Verify values are <= MAX_V
                # If L is small, max value is L+W.
                # If L is close to lcm_all, max value is L+W.
                # We can always shift the solution by k*lcm_all to fit in [1, MAX_V] if needed.
                # But since we are minimizing range, and the pattern repeats, 
                # the minimal range solution will have small L.
                # Let's check if max(result) <= MAX_V.
                if max(result) <= MAX_V:
                    val = max(result) - min(result)
                    if val < best_val:
                        best_val = val
                        best_assign = result[:]
                
                # If we found a solution for this W, it's the minimal W?
                # Yes, because we iterate W in increasing order.
                # Any solution with range R fits in a window of width R.
                # So the first W that yields a solution is the optimal range.
                # We can break both loops.
                break
        
        if best_assign is not None:
            break
    
    if best_assign is None:
        # Fallback: use the divisors themselves (valid, but not optimal)
        # We need to ensure they are distinct and <= 1000000.
        # Divisors are distinct. Max is 39.
        best_assign = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_assign]
