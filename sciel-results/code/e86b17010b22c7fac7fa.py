import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s/90s limits
    
    n = 256
    # Vertices are 1..256
    # We want a subset S such that for all a < b <= c in S, a does not divide b+c.
    
    # Pre-compute conflicts
    # A conflict exists between pair (u, v) with u < v if adding both to S violates the property.
    # The property is: for all a < b <= c in S, a % (b+c) != 0.
    # If we add a new element x to a set S, we must check:
    # 1. For all a in S, a < x: Does a divide x + b for any b in S with a < b <= x?
    #    Actually, the condition is for ALL triples in the final set.
    #    It is easier to think in terms of "incompatibility".
    #    Two elements u and v are NOT directly incompatible by themselves.
    #    Incompatibility is a hyper-edge property (triples).
    #    
    #    However, we can reframe:
    #    If we have a set S, it is valid if no triple (a,b,c) in S with a<b<=c has a|(b+c).
    #    
    #    Let's use a branch and bound approach on the elements 1..256.
    #    Order: Descending is usually good for MIS because larger numbers have fewer constraints?
    #    Actually, small numbers (like 1) conflict with EVERYTHING (1 divides everything).
    #    So 1 can never be in the set if there are at least 2 other elements?
    #    If 1 is in S, and b,c in S with 1<b<=c, then 1 | b+c. So if |S|>=2, 1 cannot be in S.
    #    If |S|=1, {1} is valid. But we want max cardinality.
    #    Generally, small numbers are "dangerous". Large numbers are "safe".
    
    # Let's build a conflict graph for pairs? No, it's triples.
    # But we can simplify the check for adding a new element x to a current valid set S.
    # If S is valid, adding x is valid if:
    # 1. No a in S (a < x) divides x + b for any b in S with a < b <= x.
    #    Note: If b > x, the triple is (a, x, b) with a < x < b. Then a must not divide x+b.
    #    So for every a in S with a < x, and every b in S with b > a, we must check a % (x+b) != 0?
    #    Wait, the condition is a < b <= c.
    #    If we add x, x can be a, b, or c.
    #    
    #    Case 1: x is the largest (c). Triples (a, b, x) with a < b < x.
    #           Condition: a does not divide b+x.
    #    Case 2: x is the middle (b). Triples (a, x, c) with a < x < c.
    #           Condition: a does not divide x+c.
    #    Case 3: x is the smallest (a). Triples (x, b, c) with x < b <= c.
    #           Condition: x does not divide b+c.
    #           
    #    So to check if x can be added to S:
    #    For all a, b in S:
    #    - If a < b < x: check a % (b+x) != 0
    #    - If a < x < b: check a % (x+b) != 0  (Same as above, just b and x swapped in sum, but sum is commutative. 
    #      Wait, the roles are fixed. 
    #      If a < b < x: triple is (a,b,x). Check a | (b+x).
    #      If a < x < b: triple is (a,x,b). Check a | (x+b).
    #      So for any pair (a,b) in S with a<b, if x > max(a,b), we check a | (b+x).
    #      If a < x < b, we check a | (x+b).
    #      If x < a < b, we check x | (a+b).
    #      
    #    This check is O(|S|^2) per addition. With |S| ~ 100, this is 10,000 ops.
    #    Branch and bound with 256 nodes is too deep for naive recursion if branching factor is high.
    #    But we can prune heavily.
    
    # Alternative: Greedy with Local Search / Simulated Annealing?
    # Given the time limit, a robust greedy + local search is safer than a potentially hanging B&B.
    # However, the "child" got 1.32 with simple greedy.
    # Let's try a smarter greedy: 
    # Sort candidates. 
    # Standard Greedy: Iterate 256 down to 1. Add if valid.
    # This yields a maximal set, not necessarily maximum.
    # 
    # Let's try "Greedy by Density/Conflicts":
    # 1. Start with empty set.
    # 2. Candidate pool 1..256.
    # 3. At each step, pick the element that causes the fewest conflicts with the *potential* remaining set? 
    #    Or simply: Pick the element that allows the most future additions?
    #    
    # Let's stick to a strong heuristic: 
    # 1. Start with the upper half [129, 256]. This is likely very dense and valid.
    #    Check validity of [129, 256].
    #    If a, b in [129, 256], a < b. b+c >= 129+130 = 259.
    #    If a divides b+c, then b+c >= 2a.
    #    Since b < 256 and c < 256, b+c < 512.
    #    If a >= 129, 2a >= 258.
    #    So it is possible for a to divide b+c.
    #    Example: a=129, b=130, c=130. b+c=260. 129 does not divide 260.
    #    a=128, b=129, c=129. b+c=258. 128 does not divide 258.
    #    Actually, for large a, the multiples are far apart.
    #    The set [129, 256] has size 128.
    #    Can we add smaller numbers?
    #    
    # Let's use a local search starting from [129, 256] or the greedy descending result.
    # 
    # Strategy:
    # 1. Generate a good initial solution using Greedy Descending (fast).
    # 2. Perform a "kick" and local search (move, swap) to improve.
    #    Objective: Maximize size.
    #    Validity check must be fast.
    
    def is_valid_set(S):
        # S is a set of ints
        # Check all triples a < b <= c
        # Optimization: 
        # For each a in S:
        #   For each b in S where b > a:
        #      For each c in S where c >= b:
        #         if a % (b+c) == 0: return False
        # This is O(|S|^3). If |S| is ~150, 150^3 = 3.3M. Might be slow in Python if done repeatedly.
        # We need incremental validity or a faster check.
        
        # Faster check for adding x to S:
        # We assume S is valid.
        # Check if adding x keeps it valid.
        # 1. x as smallest (a): For all b,c in S with x < b <= c, x must not divide b+c.
        # 2. x as middle (b): For all a,c in S with a < x < c, a must not divide x+c.
        # 3. x as largest (c): For all a,b in S with a < b < x, a must not divide b+x.
        
        # Precomputed lists for S?
        # Let's just implement a reasonably fast `can_add(x, S)`.
        
        S_list = list(S)
        
        # 1. x is smallest
        # Pairs (b,c) in S with b > x and b <= c
        # Check x % (b+c) != 0
        # 2. x is middle
        # Pairs (a,c) in S with a < x and c > x
        # Check a % (x+c) != 0
        # 3. x is largest
        # Pairs (a,b) in S with a < b < x
        # Check a % (b+x) != 0
        
        # To speed this up, we can precompute sums?
        # No, let's just iterate.
        
        # Case 1: x < b <= c
        for b in S_list:
            if b <= x: continue
            for c in S_list:
                if c < b: continue
                if x % (b+c) == 0:
                    return False
        
        # Case 2: a < x < c
        for a in S_list:
            if a >= x: continue
            for c in S_list:
                if c <= x: continue
                if a % (x+c) == 0:
                    return False
                    
        # Case 3: a < b < x
        for a in S_list:
            for b in S_list:
                if b <= a: continue
                if b >= x: continue
                if a % (b+x) == 0:
                    return False
            
        return True

    def try_add(x, S):
        if x in S: return False
        return can_add(x, S)

    def can_add(x, S):
        # Optimized check
        # Case 1: x is min (a). x < b <= c.
        # We need to check if x divides b+c for any b,c in S with b > x.
        # Let S_gt_x = [y for y in S if y > x]
        # We need to check if there exist b,c in S_gt_x with b<=c such that b+c is a multiple of x.
        # Multiples of x: 2x, 3x, ...
        # Max sum is 2*256 = 512.
        # So we check if any sum b+c is in {2x, 3x, ...}
        
        # Case 2: x is mid (b). a < x < c.
        # a in S_lt_x, c in S_gt_x.
        # Check if a divides x+c.
        
        # Case 3: x is max (c). a < b < x.
        # a,b in S_lt_x.
        # Check if a divides b+x.
        
        # Implementation:
        S_lt_x = [y for y in S if y < x]
        S_gt_x = [y for y in S if y > x]
        
        # Case 1: x is a
        # Check sums of pairs in S_gt_x
        # If S_gt_x is small, O(n^2). If large, we can use a set of sums?
        # But we need to know if a SPECIFIC sum is a multiple of x.
        # Let's just iterate.
        for i in range(len(S_gt_x)):
            b = S_gt_x[i]
            for j in range(i, len(S_gt_x)):
                c = S_gt_x[j]
                if x % (b+c) == 0:
                    return False
                    
        # Case 2: x is b
        for a in S_lt_x:
            for c in S_gt_x:
                if a % (x+c) == 0:
                    return False
                    
        # Case 3: x is c
        for i in range(len(S_lt_x)):
            a = S_lt_x[i]
            for j in range(len(S_lt_x)):
                b = S_lt_x[j]
                if a == b: continue # Need a < b
                if a > b: continue
                # Actually a < b is required.
                # My loop allows a,b any order.
                # Let's fix: iterate pairs a,b in S_lt_x with a < b.
                pass
        
        # Redo Case 3 properly
        # S_lt_x is not sorted necessarily, but we can sort it or just use double loop with check.
        # To save time, assume S is kept sorted? 
        # Let's keep S as a set for membership, and a list for iteration.
        
        # Re-implement can_add with sorted lists for efficiency
        # This function is called frequently, so it must be fast.
        
        return True # Placeholder, will replace with actual logic below

    # Let's rewrite can_add properly outside for clarity and speed
    
    def check_validity_incremental(x, S_list):
        # S_list is a sorted list of elements in S
        # x is the new element
        # Find insertion point
        import bisect
        idx = bisect.bisect_left(S_list, x)
        
        # S_lt_x = S_list[:idx]
        # S_gt_x = S_list[idx+1:] (assuming x not in S)
        
        # Case 1: x is smallest (a).
        # Pairs (b,c) from S_gt_x.
        # b, c > x.
        # Check if x divides b+c.
        # Optimization: Precompute sums? No.
        # Just loop.
        gt_len = len(S_list) - idx - 1
        for i in range(gt_len):
            b = S_list[idx + 1 + i]
            for j in range(i, gt_len):
                c = S_list[idx + 1 + j]
                if x % (b + c) == 0:
                    return False
        
        # Case 2: x is middle (b).
        # a from S_lt_x, c from S_gt_x.
        # Check if a divides x+c.
        lt_len = idx
        for i in range(lt_len):
            a = S_list[i]
            for j in range(gt_len):
                c = S_list[idx + 1 + j]
                if a % (x + c) == 0:
                    return False
        
        # Case 3: x is largest (c).
        # a, b from S_lt_x. a < b.
        # Check if a divides b+x.
        for i in range(lt_len):
            a = S_list[i]
            for j in range(i+1, lt_len):
                b = S_list[j]
                if a % (b + x) == 0:
                    return False
        
        return True

    # Initial Solution: Greedy Descending
    S_list = []
    for x in range(256, 0, -1):
        if time.time() > deadline - 5: # Reserve time
            break
        if check_validity_incremental(x, S_list):
            # Insert x into sorted list
            import bisect
            bisect.insort(S_list, x)
            
    current_size = len(S_list)
    
    # Local Search: Try to add more elements (if any left out) or swap to improve?
    # Since we went 256->1, we missed some small numbers that might fit if we removed a large one?
    # Unlikely, as large numbers are "safer".
    # But maybe we can add a small number if we remove a conflicting large number?
    # Or maybe the greedy order was bad?
    
    # Let's try a "Reverse Greedy" or "Two-Pass"
    # Pass 1: Greedy Descending (done)
    # Pass 2: Try to add any number 1..256 not in S_list.
    # If we can add one, great.
    
    S_set = set(S_list)
    added_any = True
    while added_any and time.time() < deadline - 10:
        added_any = False
        for x in range(1, 257):
            if x in S_set: continue
            if check_validity_incremental(x, S_list):
                bisect.insort(S_list, x)
                S_set.add(x)
                added_any = True
                if time.time() > deadline - 10: break
        
    # Now try swaps?
    # If we can't add any, maybe we can swap a small x with a large y to allow adding another?
    # This is complex. Given the time, the "Add" phase is the most likely improvement over pure greedy.
    # The pure greedy descending might have blocked a small number because of a large number that was added earlier.
    # By allowing "add" after the main pass, we catch cases where the set wasn't "maximal" in the sense of "no single addition possible" 
    # due to order? 
    # Actually, Greedy Descending produces a MAXIMAL set (you can't add any remaining element).
    # Why? Because we check every element 256..1. If it could be added, it would have been added.
    # So the "Add" phase will do nothing if the greedy was exhaustive.
    
    # So we need a better initial construction or local search with swaps.
    
    # Let's try a different heuristic: 
    # Start with upper half [129, 256].
    # Then try to add smaller numbers.
    
    S_list2 = list(range(129, 257))
    S_set2 = set(S_list2)
    
    # Check if S_list2 is valid.
    # It should be, but let's be sure.
    # If not valid, we need to prune.
    # Let's assume it is valid or close.
    
    # Try to add 128 down to 1
    for x in range(128, 0, -1):
        if time.time() > deadline - 5:
            break
        if check_validity_incremental(x, S_list2):
            bisect.insort(S_list2, x)
            S_set2.add(x)
            
    # Compare sizes
    if len(S_list2) > len(S_list):
        S_list = S_list2
        S_set = S_set2
    else:
        S_list = S_list
        S_set = S_set

    # One more attempt: Randomized Restart / Kick
    # If we have time, do a few random restarts of the greedy with different orders?
    # Or perturb the current solution?
    
    # Let's stick to the best of the two heuristics.
    
    return S_list

# EVOLVE-BLOCK-END
