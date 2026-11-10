"""
Strategy: Greedy Backtracking with Exact Rational Arithmetic.
The seed program returns a fixed list of 9 elements.
This program searches for a larger subset by greedily adding elements
starting from the smallest integers (1, 2, 3...), which generally
maximize the minimum distance between subset sums, making it easier
to satisfy the distinctness constraint.
"""

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    # The problem asks for a subset of [1, 22].
    # We want to maximize cardinality.
    # A set S is "dissociated" in reciprocal space if all subset sums of {1/x for x in S} are distinct.
    
    # Greedy heuristic: 
    # Smaller numbers have larger reciprocals. 
    # Including small numbers early creates large "gaps" in the sum space,
    # which reduces the chance that a sum of larger reciprocals equals a sum of smaller ones.
    # Therefore, we prioritize trying to include 1, 2, 3, etc.
    
    candidates = list(range(1, 23))
    
    # We will build the set incrementally.
    # State: current set of indices, current set of valid subset sums (as Fractions)
    
    best_set = []
    
    # We try to build a set of size k, starting from high k down to 1.
    # However, greedy construction is faster. Let's try to add elements one by one.
    
    current_set = []
    current_sums = {Fraction(0, 1)}
    
    # To make the greedy choice effective, we iterate through candidates.
    # If we can add candidate c to current_set without creating duplicate sums, we add it.
    # If we can't, we skip it? No, that's not optimal. 
    # But for a heuristic within time limits, a simple greedy "first fit" or "best fit" 
    # might get stuck. 
    # 
    # Better approach: 
    # Since the space of subsets of [1..22] is large, but we just need ONE valid set with max size,
    # we can use a backtracking search that tries to include elements in increasing order.
    # Why increasing? Because 1/1 = 1.0 is huge. 1/2 = 0.5. 
    # If 1 is in the set, any sum involving 1 is >= 1.0. Any sum not involving 1 is < 1.0 (since max sum of 1/2..1/22 is < 1).
    # This property (linear independence over reals if sorted such that next term < sum of all remaining) is strong.
    # Actually, if we pick elements such that 1/a_k < sum_{j>k} 1/a_j, we are in "superincreasing" territory backwards?
    # No, for distinct subset sums, it's sufficient that 1/a_k > sum_{j>k} 1/a_j.
    # If we pick a_k such that 1/a_k > sum of reciprocals of all larger elements, then all subset sums are distinct.
    # Let's check if we can find a long sequence where 1/a_k > sum_{j=k+1..n} 1/a_j.
    
    # Let's try to construct a sequence a_1 < a_2 < ... < a_m
    # such that 1/a_k > sum_{j=k+1}^m 1/a_j.
    # This guarantees distinctness.
    
    # We want to maximize m.
    # To maximize m, we should pick a_k as small as possible at each step?
    # No, a_k must be greater than a_{k-1}.
    # Also, the condition is on the tail sum.
    
    # Let's try a greedy algorithm from the LARGEST elements downwards?
    # Or from smallest upwards?
    
    # If we pick a_1=1, a_2=2, a_3=4, a_4=8... (powers of 2), then 1/a_k = 2 * 1/a_{k+1}.
    # Sum of tail: 1/4 + 1/8 + ... = 1/2 = 1/a_2.
    # So 1/a_k = sum of tail. This leads to collisions? 
    # If 1/a_k = sum_{j>k} 1/a_j, then the subset {a_k} has the same sum as {a_{k+1}, ..., a_m}.
    # So we need STRICT inequality: 1/a_k > sum_{j>k} 1/a_j.
    
    # Let's try to find the longest sequence a_1 < a_2 < ... < a_m in [1, 22]
    # satisfying 1/a_k > sum_{j=k+1}^m 1/a_j for all k.
    
    # This is a specific structural constraint that guarantees validity.
    # Is it necessary? No, but it's sufficient and easy to search.
    
    # Search strategy:
    # We want to maximize m.
    # We iterate m from 22 down to 1.
    # For a given m, we try to find a sequence a_1 < ... < a_m.
    # To satisfy the condition, the "tail sum" must be small.
    # This suggests picking large numbers for the tail (large a_j -> small 1/a_j).
    # So a_m, a_{m-1}, ... should be as large as possible?
    # But they must be distinct and in [1, 22].
    
    # Let's try a backtracking search from the end of the sequence (largest indices) to the start.
    # Actually, the condition couples a_k with all subsequent a_j.
    
    # Alternative: Just do a randomized hill climbing / local search on the set of elements.
    # Start with the seed [1, 2, 3, 5, 7, 11, 13, 17, 19] (size 9).
    # Try to add elements from the remaining pool [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22].
    # Check if adding an element preserves distinctness.
    # If multiple can be added, pick the one that seems "safest" (e.g., smallest reciprocal? largest reciprocal?).
    # Smallest reciprocal (largest number) adds the least "interference" to existing sums?
    # Actually, adding a small number (large reciprocal) creates large jumps, which is good for separation.
    # Adding a large number (small reciprocal) creates small adjustments.
    
    # Let's try a Greedy Additive approach:
    # 1. Start with empty set.
    # 2. Iterate candidates in increasing order of value (1, 2, 3...).
    # 3. If adding candidate c to current set results in all distinct subset sums, add it.
    # 4. Repeat until all candidates tried.
    
    # Why increasing order?
    # If we add 1, sums are {0, 1}.
    # If we add 2, sums are {0, 1, 0.5, 1.5}. Distinct.
    # If we add 3, sums are {0, 1, 0.5, 1.5, 0.33, 1.33, 0.83, 1.83}. Distinct.
    # This works well because large reciprocals dominate.
    
    # Let's implement this simple greedy first. It's fast and might find size > 9.
    
    def check_distinct(sums):
        # sums is a set of Fractions. If len(sums) == 2**len(set), it's distinct.
        # We maintain the set of sums, so if we generate 2**k sums and the set size is 2**k, it's good.
        pass

    current_set = []
    current_sums = {Fraction(0, 1)}
    
    # Candidates: 1 to 22
    for c in range(1, 23):
        if time.time() > deadline:
            break
            
        # Calculate new sums if we add 1/c
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + Fraction(1, c))
        
        # If all sums are distinct, the size of new_sums should be 2 * len(current_sums)
        if len(new_sums) == 2 * len(current_sums):
            current_set.append(c)
            current_sums = new_sums
        else:
            # Cannot add c
            pass
            
    # This greedy approach might get stuck. 
    # Example: If we have {1, 2, 3}, can we add 4?
    # Sums of {1,2,3}: 0, 1, .5, 1.5, .33, 1.33, .83, 1.83
    # Add 1/4 = .25.
    # New sums: old + old+.25.
    # Are there collisions?
    # 1.33 (1+1/3) vs 1.5833 (1+1/3+1/4)? No.
    # 0.83 (1/2+1/3) vs 1.0833 (1/2+1/3+1/4)? No.
    # It seems adding smaller numbers first is very robust.
    
    # However, what if the greedy choice of "smallest available" blocks a larger set later?
    # E.g. maybe {1, 2, 4, 8...} allows more elements than {1, 2, 3...}?
    # {1, 2, 4, 8, 16} size 5.
    # {1, 2, 3, 5, 7...} might go longer.
    
    # Let's try to improve the greedy by looking ahead or using a different order.
    # But given the 60s limit, a simple greedy is a strong baseline improvement over the static 9.
    
    # Let's also try a second pass: try to add elements in *decreasing* order?
    # No, large reciprocals (small numbers) are better for separation.
    
    # Let's try a randomized restart with the greedy heuristic.
    # We'll run the greedy multiple times with slight variations in candidate order?
    # Or just stick with the standard greedy (1 to 22).
    
    # Let's verify the result of the standard greedy.
    # If it yields size > 9, we return it.
    # If not, we return the seed.
    
    result_size = len(current_set)
    
    # If greedy didn't beat the seed, try a simple local search?
    # Actually, let's just return the greedy result if it's valid and non-empty.
    # The task says "Maximize cardinality". The seed is 9.
    # If greedy finds 10, 11, etc., that's a win.
    
    # One optimization: The check `len(new_sums) == 2 * len(current_sums)` is O(2^k).
    # For k=15, 2^15 = 32768. This is very fast.
    # For k=20, 2^20 = 1M. Still fast in Python with Fractions? Maybe slower.
    # But we stop adding when we hit a collision or run out of candidates.
    
    # Let's run the greedy.
    
    # Re-initialize to be sure
    current_set = []
    current_sums = {Fraction(0, 1)}
    
    for c in range(1, 23):
        if time.time() > deadline:
            break
        # Optimization: if current size is already high, checking is expensive.
        # But 22 is small.
        
        # Calculate new sums
        # Using a list comprehension or loop is fine.
        added = Fraction(1, c)
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + added)
            
        if len(new_sums) == len(current_sums) * 2:
            current_set.append(c)
            current_sums = new_sums
            
    # Return the best set found
    if current_set:
        return current_set
    else:
        return [1] # Fallback, though unlikely

# EVOLVE-BLOCK-END
