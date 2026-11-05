"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    time_limit = 55  # seconds, leave buffer for evaluation overhead
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(0) for _ in range(n+1)]
    for i in range(1, n+1):
        reciprocals[i] = Fraction(1, i)
    
    # For branch and bound, we need a quick upper bound.
    # Precompute suffix sums of reciprocals for fast upper bound calculation.
    # suffix_sum[i] = sum of 1/j for j from i to n
    suffix_sum = [Fraction(0) for _ in range(n+2)]
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + reciprocals[i]
    
    # Precompute for each x, the list of (a, c) pairs where a < x < c and a + c = 2*x
    # i.e., x is the midpoint. If both a and c are selected, x cannot be selected.
    # Also precompute for each x, the pairs (a, c) where x is the first or last element.
    
    # For the branch and bound, we process elements in order 1, 2, ..., n.
    # When we decide to include x, we need to:
    # 1. Check that no 3-AP is formed with two already-selected elements
    # 2. Forbid future elements that would form a 3-AP with x and an already-selected element
    
    # Precompute: for each x, and for each a < x, what is c = 2*x - a?
    # If c > x and c <= n, then selecting x and a together forbids c.
    # Similarly, for each x, and for each c > x, what is a = 2*x - c?
    # If a < x and a >= 1, then selecting x and c together forbids a.
    # But since we process in increasing order, when we select x, all previously selected
    # elements are < x. So we only need to consider:
    # - For each a in selected (a < x): c = 2*x - a. If c > x and c <= n, forbid c.
    # - Check: for each a in selected (a < x), is b = (a+x)//2 in selected? (with a+x even)
    
    # Let's precompute for each x, the set of a < x such that (a+x) is even
    # and (a+x)//2 < x (which is always true if a < x).
    
    # Actually, let's use a different representation. We'll use bitsets.
    # selected_bits: bit i is 1 if i+1 is selected (using 0-indexed bits for numbers 1..n)
    # forbidden_bits: bit i is 1 if i+1 is forbidden
    
    # Precompute for each x (1-indexed), the bitset of numbers a < x such that
    # (a + x) is even. These are the numbers that, if selected along with x, 
    # would create a midpoint b = (a+x)//2 that must also be checked.
    
    # Actually, let's think about this more carefully with bitsets.
    
    # When we select x, we need to check:
    # For each a in selected with a < x:
    #   If (a + x) % 2 == 0, then b = (a+x)//2. If b is in selected, conflict.
    
    # We can precompute for each x, a bitset "midpoints_of_x" where bit a is set if
    # a < x and (a+x) is even, meaning that if both a and x are selected, 
    # then b=(a+x)//2 cannot be selected.
    
    # But actually, we check at the time of selecting x: for each a in selected with a < x
    # and (a+x) even, check if (a+x)//2 is in selected.
    
    # Let's precompute for each x, the list of (a, b) pairs where a < b < x and a + x = 2*b.
    # i.e., b = (a+x)//2, and a + x is even.
    
    # For efficiency, let's precompute:
    # conflict_pairs[x] = list of (a, b) where a < b < x, a + x = 2*b
    # When selecting x, if any (a, b) in conflict_pairs[x] has both a and b in selected, conflict.
    
    # Also, when selecting x, for each a in selected with a < x:
    #   c = 2*x - a. If c > x and c <= n, add c to forbidden.
    
    # Let's precompute forbid_pairs[x] = list of (a, c) where a < x < c, a + c = 2*x
    # When selecting x, for each a in selected with a < x, if (a, c) in forbid_pairs[x] 
    # for some c > x, then c is forbidden.
    
    # Actually, it's simpler: for each a < x, c = 2*x - a. If c > x and c <= n, forbid c.
    
    # Let's precompute for each x, the set of a < x such that 2*x - a > x and 2*x - a <= n.
    # That is, a < x and 2*x - n <= a < x.
    # So a is in [max(1, 2*x - n), x-1].
    
    # For the branch and bound, let's use the following approach:
    # - Process x from 1 to n
    # - Maintain selected as a set (or bitset) and forbidden as a set (or bitset)
    # - At each step, if x is forbidden, skip it
    # - Otherwise, try selecting x (if valid) and not selecting x
    # - Use upper bound: current_sum + sum of 1/y for y > x not in forbidden
    
    # For efficiency with bitsets, let's represent selected and forbidden as integers.
    # Bit i (0-indexed) represents number i+1.
    
    # Precompute for each x (1-indexed), the bitset of a's (a < x) such that
    # selecting x along with a would forbid c = 2*x - a (if c > x and c <= n).
    # Actually, we just need to iterate over a in selected that are < x.
    
    # Let's use a recursive branch and bound with memoization on (x, selected_bits, forbidden_bits)
    # but the state space might be too large. Let's try without memoization first,
    # with aggressive pruning.
    
    # Precompute the upper bound contribution: for each x, the sum of 1/y for y from x to n.
    # This is suffix_sum[x].
    
    # For a tighter bound, we need to subtract the reciprocals of forbidden elements > x.
    # Let's precompute a function that, given forbidden_bits and x, returns the sum of
    # 1/y for y > x that are not forbidden.
    
    # Actually, for speed, let's use floating point for the upper bound and only use
    # exact arithmetic for the final answer.
    
    # Precompute float reciprocals for fast upper bound computation.
    float_reciprocals = [0.0] * (n+1)
    for i in range(1, n+1):
        float_reciprocals[i] = 1.0 / i
    
    # Precompute float suffix sums
    float_suffix = [0.0] * (n+2)
    for i in range(n, 0, -1):
        float_suffix[i] = float_suffix[i+1] + float_reciprocals[i]
    
    # Precompute for each x, the list of a < x such that (a + x) is even
    # and b = (a+x)//2 is also < x (which is always true if a < x).
    # These are the potential conflicts when adding x.
    
    # For each x, precompute the bitset of a's (a < x) where (a+x) % 2 == 0
    # and b = (a+x)//2 < x.
    # Actually, b = (a+x)//2 < x iff a < x, which is always true.
    # So for each x, the a's that matter are those with a < x and (a+x) even.
    
    # Let's precompute:
    # conflict_a_bits[x] = bitset of a in [1, x-1] where (a+x) % 2 == 0
    # When selecting x, we need to check: for each a in (selected & conflict_a_bits[x]),
    # is b = (a+x)//2 in selected?
    
    # This is still expensive. Let's try a different approach.
    
    # Alternative: precompute for each x, the list of pairs (a, b) where a < b < x
    # and a + x = 2*b. When selecting x, check if any such pair has both a and b in selected.
    
    conflict_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        # b ranges from 1 to x-1, a = 2*b - x, need a >= 1 and a < b
        # a = 2*b - x >= 1 => b >= (x+1)//2
        # a < b => 2*b - x < b => b < x
        # So b ranges from ceil((x+1)/2) to x-1
        b_min = (x + 1) // 2 + 1 if (x + 1) % 2 == 0 else (x + 1) // 2
        # Let's just iterate b from 1 to x-1 and check
        for b in range(1, x):
            a = 2*b - x
            if 1 <= a < b:
                conflict_pairs[x].append((a, b))
    
    # For each x, precompute the list of c > x such that there exists a < x with a + c = 2*x
    # i.e., c = 2*x - a, where a < x and c > x and c <= n.
    # For each a in [max(1, 2*x-n), x-1], c = 2*x - a.
    
    # When selecting x, for each a in selected with a < x, c = 2*x - a.
    # If c > x and c <= n, add c to forbidden.
    
    # Let's precompute for each x, the mapping from a to c:
    # forbid_map[x] = dict mapping a to c = 2*x - a, for a in [max(1, 2*x-n), x-1]
    
    # Actually, we can compute c on the fly. Let's just iterate over a in selected.
    
    # Now let's implement the branch and bound.
    
    best_sum = Fraction(0)
    best_set = set()
    
    def is_valid_selection(x, selected_bits):
        """Check if x can be added to the selected set without creating a 3-AP."""
        # Check conflict_pairs[x]: for each (a, b), if both a and b are in selected, conflict.
        for (a, b) in conflict_pairs[x]:
            if (selected_bits >> (a-1)) & 1 and (selected_bits >> (b-1)) & 1:
                return False
        return True
    
    def get_forbidden_from_selection(x, selected_bits):
        """Get the bitset of c > x that are forbidden if we select x."""
        forbidden = 0
        # For each a in selected with a < x:
        #   c = 2*x - a. If c > x and c <= n, set bit c-1.
        # Iterate over set bits in selected_bits that are < x.
        # For efficiency, iterate over a from 1 to x-1 and check if bit a-1 is set.
        for a in range(1, x):
            if (selected_bits >> (a-1)) & 1:
                c = 2*x - a
                if x < c <= n:
                    forbidden |= (1 << (c-1))
        return forbidden
    
    # Upper bound: current_sum + sum of 1/y for y > x that are not in forbidden_bits
    # For speed, use floating point.
    def compute_upper_bound(x, current_float_sum, forbidden_bits):
        """Compute an upper bound on the achievable sum from position x onwards."""
        # Sum of 1/y for y from x to n, minus sum of 1/y for y in forbidden_bits with y >= x.
        ub = current_float_sum + float_suffix[x]
        # Subtract forbidden elements >= x
        # Iterate over set bits in forbidden_bits that are >= x
        for y in range(x, n+1):
            if (forbidden_bits >> (y-1)) & 1:
                ub -= float_reciprocals[y]
        return ub
    
    # Global variables for the recursive search
    global_best = [Fraction(0)]
    global_best_set = [set()]
    node_count = [0]
    timeout_flag = [False]
    
    def branch_and_bound(x, selected_bits, forbidden_bits, current_sum, current_float_sum):
        """Recursively explore the search tree."""
        if timeout_flag[0]:
            return
        
        node_count[0] += 1
        if node_count[0] % 100000 == 0:
            if time.time() - start_time > time_limit:
                timeout_flag[0] = True
                return
        
        if x > n:
            # Base case: all elements processed
            if current_sum > global_best[0]:
                global_best[0] = current_sum
                # Reconstruct the set from selected_bits
                s = set()
                for i in range(1, n+1):
                    if (selected_bits >> (i-1)) & 1:
                        s.add(i)
                global_best_set[0] = s
            return
        
        # Upper bound check
        ub = compute_upper_bound(x, current_float_sum, forbidden_bits)
        if ub <= float(global_best[0]):
            return
        
        # If x is forbidden, we must skip it
        if (forbidden_bits >> (x-1)) & 1:
            branch_and_bound(x+1, selected_bits, forbidden_bits, current_sum, current_float_sum)
            return
        
        # Try selecting x first (since it has higher weight)
        if is_valid_selection(x, selected_bits):
            new_selected_bits = selected_bits | (1 << (x-1))
            new_forbidden_bits = forbidden_bits | get_forbidden_from_selection(x, selected_bits)
            branch_and_bound(x+1, new_selected_bits, new_forbidden_bits,
                           current_sum + reciprocals[x], current_float_sum + float_reciprocals[x])
        
        # Try not selecting x
        branch_and_bound(x+1, selected_bits, forbidden_bits, current_sum, current_float_sum)
    
    # Initialize with a greedy solution to get a good lower bound
    def greedy_solution():
        selected = set()
        for x in range(1, n+1):
            can_add = True
            for a in selected:
                if a < x:
                    b = (a + x) // 2
                    if (a + x) % 2 == 0 and b in selected and a < b < x:
                        can_add = False
                        break
            if can_add:
                selected.add(x)
        return selected
    
    initial_selected = greedy_solution()
    initial_sum = sum(reciprocals[x] for x in initial_selected)
    global_best[0] = initial_sum
    global_best_set[0] = initial_selected
    
    # Run branch and bound
    branch_and_bound(1, 0, 0, Fraction(0), 0.0)
    
    # Check if we timed out
    if timeout_flag[0]:
        # Return the best found so far
        return list(global_best_set[0])
    
    return list(global_best_set[0])
# EVOLVE-BLOCK-END
