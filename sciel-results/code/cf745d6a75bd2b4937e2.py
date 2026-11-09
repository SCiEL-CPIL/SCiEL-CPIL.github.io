# EVOLVE-BLOCK-START
import sys
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # DP approach:
    # dp is a set of (sum, indices_tuple) pairs.
    # We iterate through each denominator.
    # For each existing state, we can either skip the current denominator or take it.
    # If we take it, the new sum must be < 1.
    # To keep the state space manageable, we might need to prune.
    # However, with n=47, the number of distinct fractions < 1 formed by subsets of {1/2..1/48} is not too large.
    # Let's estimate. The number of distinct sums is bounded by the number of subsets, which is 2^47, but many sums are equal or close.
    # Actually, the number of distinct rational numbers with denominator dividing LCM(2..48) is huge.
    # But we are summing *distinct* unit fractions.
    # A better way: Use a dictionary mapping sum -> indices.
    # We want to keep the "best" (largest sum) for each unique sum? No, different indices might lead to different future possibilities?
    # Actually, since we only care about the final sum being < 1 and maximizing it, and the items are processed in a fixed order,
    # the "future" depends only on the current sum and the current index.
    # So, for a given index i and a given sum S, the set of achievable final sums from i+1..n-1 is the same regardless of which indices were chosen to form S.
    # Therefore, for each i, we only need to keep the best (largest) sum for each "state"?
    # No, we need to keep all distinct sums because a smaller sum might allow adding a larger fraction later?
    # Wait, if we have two states at step i: (S1, I1) and (S2, I2) with S1 < S2.
    # Any set of future items J can be added to both.
    # S1 + sum(J) < 1 implies S2 + sum(J) < 1 is not necessarily true, but S1+sum(J) < S2+sum(J).
    # If we maximize the final sum, and S2 > S1, then for any J that is valid for S1, S2+J is also valid (if S2+J < 1) and larger.
    # If S2+J >= 1, then J is invalid for S2 but might be valid for S1.
    # So we cannot simply discard S1. We need to keep all distinct sums.
    
    # However, the number of distinct sums might explode.
    # Let's try a different DP:
    # Let dp[i][s] be True if sum s is achievable using a subset of the first i denominators.
    # This is still hard because s is a Fraction.
    
    # Alternative: Meet in the Middle.
    # Split denominators into two halves: A (first 23) and B (last 24).
    # Generate all subset sums for A: list of (sum, indices). Filter sum < 1.
    # Generate all subset sums for B: list of (sum, indices). Filter sum < 1.
    # Sort B by sum.
    # For each sum_a in A, we want to find the largest sum_b in B such that sum_a + sum_b < 1.
    # This is a standard "closest pair" or binary search problem.
    # Number of subsets for 23 items is 2^23 = 8,388,608. This is a bit large for Python to generate and store all tuples with indices, but maybe feasible with optimization?
    # 8M items * (Fraction + Tuple) is heavy.
    
    # Let's stick to Beam Search but improve it? The directive says "do not reuse current designs".
    # So I must use a new approach.
    
    # Let's try DP with pruning.
    # Key insight: The sums are rational.
    # We can store the best (max sum) for each "residue" or just keep a set of sums.
    # Let's try generating all subset sums for the first half and second half.
    # 2^23 is ~8.4M. In Python, this might be slow or memory intensive.
    # 2^24 is ~16M.
    
    # Maybe a smarter DP:
    # We process items one by one.
    # We maintain a set of achievable sums.
    # At each step, we double the set size (roughly).
    # But we can prune sums that are "dominated".
    # A sum S1 is dominated by S2 if S1 < S2 and S1 cannot lead to a better final result.
    # As discussed, S1 is not dominated by S2 if S2 is too large to allow some future additions that S1 can.
    # However, if we process items in descending order of value (ascending denominators), the future items are smaller.
    # If we have processed large items, the remaining items are small.
    # Let's sort denominators in descending order of fraction value (i.e., ascending denominator).
    # d = 2, 3, ..., 48.
    # If we have a sum S, the maximum we can add from the remaining items is sum(remaining).
    # If S + sum(remaining) < 1, then any subset of remaining is valid. We just take the one that maximizes the sum?
    # No, we want to maximize the total sum.
    
    # Let's go back to Meet in the Middle but optimize the storage.
    # We only need the sum and the indices.
    # For half A (first 23 items), generate all subsets.
    # For half B (last 24 items), generate all subsets.
    
    # To speed up generation, we can use iterative expansion.
    
    def gen_subsets(items):
        # items: list of (value, index)
        # returns list of (sum, indices_tuple)
        subsets = [(Fraction(0), ())]
        for val, idx in items:
            new_subsets = []
            for s, inds in subsets:
                new_s = s + val
                if new_s < 1:
                    new_subsets.append((new_s, inds + (idx,)))
            subsets.extend(new_subsets)
        return subsets

    # 2^23 is 8.4M. Extending a list of 8M items in a loop of 23...
    # The list size doubles each time.
    # Total operations ~ 2^24.
    # Python loop of 16M iterations might take a few seconds.
    # Storing 8M tuples of (Fraction, tuple) is memory heavy.
    # Fraction objects are large.
    
    # Is there a more efficient DP?
    # Since n=47 is small, maybe a simple recursive search with memoization on (i, current_sum) is too slow due to Fraction keys.
    
    # Let's try a different angle:
    # The deficit is 1 - sum. We want to minimize it.
    # This is equivalent to maximizing sum.
    
    # What if we use a priority queue (Best-First Search)?
    # State: (sum, current_index, indices_tuple)
    # Start: (0, 0, ())
    # At each step, we can either take item i or skip item i.
    # We prioritize states with higher sum.
    # We stop when we pop a state that has processed all items?
    # No, we need to explore.
    # But if we always expand the state with the highest sum, the first complete state we find is the optimal one.
    # This is Dijkstra's algorithm on a DAG (actually a tree of decisions).
    # Since the graph is a tree, and we use a max-heap, this is essentially Branch and Bound or Best-First Search.
    # Is it faster than Beam Search?
    # Beam Search keeps the top K. Best-First keeps the top 1 path? No, it keeps a heap of all active states.
    # The heap can grow large.
    
    # However, for "maximize sum < 1", Best-First Search with a Max-Heap is very effective.
    # The first time we reach the end (i == n), we have the optimal solution.
    # Why? Because any other path in the heap has a sum <= the current best sum (since we always pop the max).
    # And any extension of a state with sum S' <= S_best will have sum <= S_best + (sum of remaining items).
    # Wait, this logic holds if we know that S_best is achievable.
    # Yes, if we pop a state (S, i, inds) where i == n, then S is a valid sum.
    # Any other state in the heap has sum <= S (because it's a max-heap).
    # Any descendant of any state in the heap will have sum <= (state sum) + (sum of remaining items).
    # This doesn't immediately prove optimality unless we prune.
    # But since we are looking for the *maximum* sum, and we always expand the state with the *current* maximum sum,
    # if we find a complete state with sum S, is it possible that a state in the heap with sum S' < S could lead to S' + delta > S?
    # Yes, if delta is large.
    # So simple Best-First is not guaranteed to find the global optimum first.
    
    # However, if we use A* or similar?
    # Heuristic: sum of remaining items.
    # Priority = current_sum + sum_of_remaining[i].
    # This is an upper bound on the final sum.
    # We want to maximize this upper bound.
    # If we pop a state with i == n, its upper bound is its actual sum.
    # If the upper bound of the popped state is less than the best found so far, we can stop?
    # No, we want to find the best.
    
    # Let's stick to the DP/Meet-in-the-Middle but optimize it.
    # Maybe the number of distinct sums is much smaller than 2^23?
    # Let's test the size of the set of sums.
    
    def construct():
        from fractions import Fraction
        
        denominators = list(range(2, 49))
        n = len(denominators)
        fracs = [Fraction(1, d) for d in denominators]
        
        # Meet in the Middle
        # Split into two halves
        mid = n // 2
        left_denoms = denominators[:mid]
        right_denoms = denominators[mid:]
        
        left_fracs = fracs[:mid]
        right_fracs = fracs[mid:]
        
        def gen_sums(fracs, start_idx):
            # Returns list of (sum, indices_tuple)
            # indices are relative to the original list?
            # It's easier to keep global indices.
            sums = [(Fraction(0), ())]
            for i, f in enumerate(fracs):
                global_idx = start_idx + i
                new_sums = []
                for s, inds in sums:
                    new_s = s + f
                    if new_s < 1:
                        new_sums.append((new_s, inds + (global_idx,)))
                sums.extend(new_sums)
            return sums
        
        left_sums = gen_sums(left_fracs, 0)
        right_sums = gen_sums(right_fracs, mid)
        
        # Sort right_sums by sum
        right_sums.sort(key=lambda x: x[0])
        
        # Extract right sums for binary search
        right_values = [s[0] for s in right_sums]
        
        best_sum = Fraction(0)
        best_inds = ()
        
        import bisect
        
        for s_l, inds_l in left_sums:
            # We want max s_r such that s_l + s_r < 1
            # s_r < 1 - s_l
            target = 1 - s_l
            # Find rightmost index in right_values where value < target
            # bisect_left finds the first index where value >= target.
            # So the index before that is the largest value < target.
            idx = bisect.bisect_left(right_values, target)
            if idx > 0:
                s_r, inds_r = right_sums[idx - 1]
                total_sum = s_l + s_r
                if total_sum > best_sum:
                    best_sum = total_sum
                    best_inds = inds_l + inds_r
        
        return list(best_inds)
# EVOLVE-BLOCK-END
