def construct():
    """Construct a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.
    
    Uses optimized backtracking with pruning to find a maximal Sidon set.
    """
    import sys
    sys.setrecursionlimit(10000)
    
    # Required elements
    required = [1, 2, 5, 11]
    # Range
    N = 284
    
    # Precompute potential conflicts? 
    # It's better to maintain the set of used sums dynamically.
    
    used_sums = set()
    # Initialize with required elements
    for i in range(len(required)):
        for j in range(i, len(required)):
            used_sums.add(required[i] + required[j])
            
    current_set = sorted(required)
    
    # Candidates are 12 to 284
    # We want to add as many as possible.
    
    # To speed up, we can sort candidates.
    # A good heuristic for Sidon sets is to pick larger numbers earlier? 
    # Or smaller? 
    # Usually, dense sets use smaller numbers. 
    # But since 1, 2, 5, 11 are small, we are constrained.
    
    def backtrack(start_idx, current_set, used_sums):
        """
        Recursively try to add elements from the candidate pool.
        start_idx: index in the candidate list to start checking from.
        current_set: sorted list of selected elements.
        used_sums: set of all pairwise sums.
        """
        if not candidates:
            return current_set[:]
        
        # Heuristic: Try candidates in a specific order.
        # We can try to add the largest possible candidate first to leave room for smaller ones?
        # Or smallest to pack densely?
        # For Sidon sets, density is key. 
        # Let's try candidates in ascending order, but skip those that conflict.
        
        best_solution = current_set[:]
        
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Check if x can be added
            # New sums: x + y for y in current_set, and x + x
            new_sums = set()
            valid = True
            for y in current_set:
                s = x + y
                if s in used_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s = 2 * x
                if s in used_sums or s in new_sums: # s in new_sums is only if x in current_set, which is false
                    valid = False
                else:
                    new_sums.add(s)
            
            if valid:
                # Add x
                new_set = current_set + [x]
                new_set.sort()
                new_used_sums = used_sums.union(new_sums)
                
                # Recurse
                # We pass i+1 to ensure we only pick larger indices? 
                # No, we can pick any unused candidate. 
                # But to avoid permutations, we should enforce an order.
                # Since we are picking from a fixed list `candidates`, we can just recurse with the next index.
                # However, the "best" set might not use the next candidate.
                # Standard backtracking: try to include x, then try to exclude x.
                
                # Include x
                sol = backtrack(i + 1, new_set, new_used_sums)
                if len(sol) > len(best_solution):
                    best_solution = sol
                
                # Exclude x (implicitly handled by the loop moving to i+1)
                
        return best_solution

    # Candidate list
    candidates = list(range(12, N + 1))
    
    # Optimization: 
    # The simple backtracking above is O(N^k) which is too slow for N=272, k=16.
    # We need strong pruning.
    
    # Better approach: 
    # 1. Greedy construction to get a good initial set.
    # 2. Local search / Simulated Annealing to improve it.
    # The current program does this but is slow.
    
    # Let's implement a faster version of the current approach.
    
    def compute_sums(arr):
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                sums.add(arr[i] + arr[j])
        return sums

    def can_add(arr, used_sums, x):
        """Check if x can be added to arr without conflict."""
        # Check x + y for all y in arr
        for y in arr:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        # Check if x+y1 == x+y2 is possible? No, y1 != y2 implies x+y1 != x+y2.
        # Check if x+y == 2x? y=x, but x not in arr.
        # So just checking against used_sums is sufficient?
        # Wait. If we add x, the new sums are {x+y | y in arr} U {2x}.
        # We need these new sums to be distinct from each other and from used_sums.
        # Distinct from each other: 
        # x+y1 == x+y2 => y1==y2 (impossible for distinct y in arr)
        # x+y == 2x => y=x (impossible as x not in arr)
        # So we just need {x+y | y in arr} U {2x} to be disjoint from used_sums.
        
        new_sums = set()
        for y in arr:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        
        if new_sums & used_sums:
            return False
        return True

    def greedy_construct():
        a = [1, 2, 5, 11]
        sums = compute_sums(a)
        candidates = list(range(12, 285))
        
        # Sort candidates? 
        # Heuristic: Pick the candidate that blocks the fewest other candidates.
        
        while True:
            best_x = None
            min_conflicts = float('inf')
            
            for x in candidates:
                if not can_add(a, sums, x):
                    continue
                
                # Count how many other candidates become invalid if we add x
                # This is expensive. 
                # Approximation: Prefer smaller x? Or larger x?
                # For Sidon sets, larger numbers are "safer" because they create larger sums 
                # which are less likely to conflict with small sums from the seed.
                # However, we want to maximize size. 
                
                # Let's just pick the smallest valid x first (greedy)
                # Or the one with lowest "cost".
                
                if best_x is None or x < best_x:
                    best_x = x
                    min_conflicts = 0 # Placeholder
                    
            if best_x is None:
                break
                
            a.append(best_x)
            sums = compute_sums(a)
            candidates.remove(best_x)
            a.sort()
            
        return a

    # Greedy is often suboptimal. 
    # Let's use Randomized Greedy + Local Search.
    
    import random
    random.seed(42)
    
    def randomized_greedy():
        a = [1, 2, 5, 11]
        sums = compute_sums(a)
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(a, sums, x):
                a.append(x)
                sums = compute_sums(a)
                a.sort()
                
        return a

    def local_search(arr, iterations=1000):
        """
        Try to improve the set by swapping elements or adding/removing.
        """
        current = sorted(arr)
        best = current[:]
        best_size = len(current)
        
        for _ in range(iterations):
            improved = False
            
            # 1. Try to add a new element
            sums = compute_sums(current)
            for x in range(1, 285):
                if x in current:
                    continue
                if can_add(current, sums, x):
                    new_arr = sorted(current + [x])
                    if len(new_arr) > best_size:
                        best = new_arr[:]
                        best_size = len(new_arr)
                    current = new_arr
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try to swap an element
            # Pick a random non-required element
            non_required = [x for x in current if x not in [1, 2, 5, 11]]
            if not non_required:
                break
                
            idx = random.randint(0, len(non_required) - 1)
            old_val = non_required[idx]
            
            # Try to replace old_val with a new value
            # To speed up, we can try values near old_val or random ones.
            # Let's try a few random candidates.
            found_swap = False
            for _ in range(50): # Try 50 random candidates
                new_val = random.randint(1, 284)
                if new_val in current or new_val == old_val:
                    continue
                if new_val in [1, 2, 5, 11]:
                    continue
                    
                test_arr = current[:]
                test_arr.remove(old_val)
                test_arr.append(new_val)
                test_arr.sort()
                
                if is_valid(test_arr):
                    # Accept if it doesn't reduce size (it doesn't, it's a swap)
                    # But does it allow future additions? 
                    # We just accept the swap.
                    current = test_arr
                    found_swap = True
                    break
            
            if not found_swap:
                # If no swap found, maybe remove an element to allow a different configuration?
                # Or just stop.
                pass
                
        return best

    def is_valid(arr):
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Run multiple randomized greedy attempts
    best_result = []
    for _ in range(20):
        candidate = randomized_greedy()
        improved = local_search(candidate, iterations=200)
        if len(improved) > len(best_result):
            best_result = improved

    # Final check
    if not all(x in best_result for x in [1, 2, 5, 11]):
        return [1, 2, 5, 11]
        
    return sorted(best_result)
