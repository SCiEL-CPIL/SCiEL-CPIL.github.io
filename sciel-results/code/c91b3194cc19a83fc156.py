"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _subset_sums_unique(seq):
    """Check that all subset sums of seq are distinct."""
    n = len(seq)
    total_subsets = 1 << n
    # Use a set to track seen sums
    seen = set()
    # Precompute subset sums using iterative doubling
    # For n=12, 4096 subsets is fast
    sums = [0] * total_subsets
    for mask in range(1, total_subsets):
        # Find lowest set bit
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        sums[mask] = sums[prev] + seq[idx]
        if sums[mask] in seen:
            return False
        seen.add(sums[mask])
    return True

def _try_construct(greedy_order, max_val=1048576):
    """
    Try to construct a valid set using a greedy approach with a specific order.
    We pick elements one by one. At each step, we try the smallest possible value
    that maintains the distinct subset sum property.
    """
    current_set = []
    # We want to minimize the max element.
    # Greedy: for each position, try smallest integer > 0 that works.
    # But we must ensure we can complete the set.
    # Simple greedy might get stuck. We can use backtracking or just greedy and hope.
    # Given the constraint is tight (n=12), simple greedy with small steps might work.
    
    # To be efficient, we can maintain the set of existing subset sums.
    # When adding a new element x, the new sums are S_old U (S_old + x).
    # They must be disjoint.
    
    sums_set = {0}
    sums_list = [0]
    
    for i in range(12):
        # Try values starting from 1
        # Optimization: if we have k elements, the max sum is sum(current_set).
        # The new element x must be such that (sums_list + x) doesn't intersect sums_list.
        # This implies x must be > max(sums_list) OR x must be in the gaps.
        # For distinct subset sums, a sufficient condition is x > sum(current_set).
        # This leads to powers of 2.
        # Can we do better? Yes, if there are gaps.
        
        found = False
        # We don't know the optimal x in advance.
        # Let's try small values first.
        # However, if we pick x too small, we might fill gaps and make future x's large.
        # If we pick x too large, we increase the max immediately.
        
        # Heuristic: try x from 1 upwards, but skip if x <= max(sums_list) and causes collision.
        # Actually, if x <= max(sums_list), we must check carefully.
        # If x > max(sums_list), it's always safe (for distinctness relative to current set).
        
        # Let's iterate x from 1 to some limit.
        # The limit can be the current best max found in a previous run, or just 1048576.
        # To keep it fast, we can stop if x exceeds some heuristic bound.
        
        # A better strategy for minimization:
        # We want the final max to be small.
        # This suggests keeping all elements small.
        # Let's try x from 1 to 2048 (since 2048 is the known optimal max).
        # If we find a valid set with max < 2048, we win.
        
        # But we must complete the set.
        # Greedy might fail if it picks a value that makes completion impossible.
        # For n=12, the search space is small enough that a simple greedy with restarts
        # or a slightly smarter heuristic might find the optimal or near-optimal.
        
        # Let's try a simple greedy: pick the smallest x that maintains validity.
        # If this results in max > 2048, we fail to improve.
        # If it results in max <= 2048, we check if it's better.
        
        # To avoid O(N^2) checks for each x, we can use the property:
        # If x > sum(current_set), it's valid.
        # If x <= sum(current_set), we check intersection.
        
        # Let's just iterate x.
        # To ensure we find the global min max, we might need to be smarter.
        # But the prompt says "Minimize the largest integer".
        # The binary sequence has max 2048.
        # Is it possible to have max < 2048?
        # The number of subset sums is 2^n = 4096.
        # The sums range from 0 to Sum.
        # If all sums are distinct, Sum >= 4095.
        # Sum = sum(a_i).
        # If max(a_i) = M, then Sum <= 12*M.
        # So 12*M >= 4095 => M >= 341.25.
        # So M >= 342 is the absolute lower bound.
        # 2048 is much larger. So there is room for improvement.
        
        # We need a search algorithm that finds a set with small max.
        # Greedy smallest-first:
        # 1. a1=1. Sums: {0,1}. Max=1.
        # 2. a2=2? Sums: {0,1,2,3}. Max=2.
        # 3. a3=4? Sums: {0..7}. Max=4.
        # This leads to powers of 2.
        
        # Can we deviate?
        # a1=1, a2=2. Sums: {0,1,2,3}.
        # Try a3=3? Sums: {0,1,2,3} U {3,4,5,6} = {0..6}. Distinct? Yes.
        # Wait, 3 is in Sums_old. 3+0=3. Collision?
        # Subset {3} sum=3. Subset {1,2} sum=3. Collision!
        # So a3 cannot be 3 if {1,2} is present.
        # a3=4 works.
        
        # What if a2=3?
        # a1=1. Sums {0,1}.
        # a2=3. Sums {0,1,3,4}.
        # a3?
        # Try 2. Sums {0,1,3,4} U {2,3,5,6}.
        # Collision: 3 (from a2) and 3 (from 1+2).
        # Try 4. Sums {0,1,3,4} U {4,5,7,8}.
        # Collision: 4 (from a2) and 4 (from 1+3? No, 1+3=4 is in old sums? Yes, 4 is in old sums).
        # Wait, old sums: 0, 1, 3, 4.
        # New sums with x=4: 4, 5, 7, 8.
        # 4 is in old sums. Collision.
        # Try 5. Old: 0,1,3,4. New: 5,6,8,9.
        # Intersection? No.
        # So {1,3,5} works. Max=5.
        # Sums: 0,1,3,4,5,6,8,9.
        
        # Continue with {1,3,5}.
        # Try a4=2?
        # Old sums: 0,1,3,4,5,6,8,9.
        # New sums with 2: 2,3,5,6,7,8,10,11.
        # Collision: 3 (old), 3 (new).
        # Try a4=4?
        # New sums: 4,5,7,8,9,10,12,13.
        # Collision: 4 (old).
        # Try a4=7?
        # New sums: 7,8,10,11,12,13,15,16.
        # Collision: 8 (old).
        # Try a4=8?
        # New sums: 8,9,11,12,13,14,16,17.
        # Collision: 8, 9.
        
        # This suggests we need a more systematic search.
        # Since n=12 is small, we can use a DFS with pruning.
        # State: current set, current sums, current max.
        # Pruning: if current max > best_max, prune.
        # Also, if we can't fill the remaining slots without exceeding best_max, prune.
        
        # However, 60s limit.
        # A simple DFS might be too slow if not pruned well.
        # But the space is small.
        
        # Let's implement a DFS that tries to minimize the max element.
        # We sort the elements we pick to ensure we explore smaller maxes first?
        # No, the order of picking doesn't matter for the set, but it matters for the search.
        # We can enforce a1 <= a2 <= ... <= an.
        # Then a_n is the max.
        # We want to minimize a_n.
        
        # We can iterate a_n from lower_bound upwards.
        # For a fixed a_n, can we find a1..a_{n-1} <= a_n such that all subset sums are distinct?
        # This is a constraint satisfaction problem.
        
        # Given the time limit, let's stick to a robust greedy or limited backtracking.
        # The binary sequence is a safe fallback.
        
        # Let's try to find a better one using a simple heuristic search.
        # We will try to construct the set element by element, picking the smallest valid element.
        # If we get stuck or exceed a certain max, we backtrack.
        
        pass

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    # Fallback: Powers of 2
    best_set = [1 << i for i in range(12)]
    best_max = max(best_set)
    
    # Try to improve using a DFS with pruning
    # We assume sorted order a1 <= a2 <= ... <= a12
    # We want to minimize a12.
    
    def dfs(idx, current_set, sums_set, current_max):
        nonlocal best_set, best_max
        
        if time.time() > deadline:
            return
        
        if idx == 12:
            if current_max < best_max:
                best_max = current_max
                best_set = current_set[:]
            return
        
        # Lower bound for next element
        # The smallest possible value is current_set[-1] if we allow duplicates? No, distinct integers.
        # So min_val = current_set[-1] + 1 if idx > 0 else 1.
        # But we can pick smaller if not sorted? We enforce sorted.
        
        # Heuristic upper bound for next element to keep max small
        # If we pick x, the new max is max(current_max, x).
        # We want this to be < best_max.
        # So x < best_max.
        
        min_val = current_set[-1] + 1 if current_set else 1
        max_val = best_max - 1 # We want strictly smaller max
        
        if min_val > max_val:
            return
            
        # Try values from min_val to max_val
        # To find the smallest max quickly, we should try smaller x first.
        
        for x in range(min_val, max_val + 1):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            # New sums: sums_set U (sums_set + x)
            # Must be disjoint.
            
            # Optimization: if x > max(sums_set), it's always disjoint.
            # max_sums = max(sums_set)
            # if x > max_sums: valid
            
            # Otherwise, check intersection.
            # For speed, we can just generate new sums and check.
            # n is small, sums_set size is 2^idx.
            
            valid = True
            new_sums_set = set(sums_set)
            for s in sums_set:
                ns = s + x
                if ns in new_sums_set:
                    valid = False
                    break
                new_sums_set.add(ns)
            
            if not valid:
                continue
            
            # Recurse
            dfs(idx + 1, current_set + [x], new_sums_set, max(current_max, x))
            
            # If we found a solution, we might break?
            # No, we want to explore other x's that might lead to a smaller final max?
            # Since we iterate x from small to large, the first valid x might not lead to the global min max.
            # But pruning with best_max helps.
            
            # Actually, if we find a complete set, best_max is updated.
            # Then subsequent iterations with larger x might be pruned if x >= best_max.
            # But we are iterating x < best_max.
            # So we continue to find potentially better sets.
            
    # Start DFS
    # Initial sums: {0}
    dfs(0, [], {0}, 0)
    
    return best_set

# EVOLVE-BLOCK-END
