import time
import itertools
import random

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Uses a candidate generation and backtracking search approach.
    """
    start_time = time.time()
    time_limit = 110
    
    def get_diffs(row):
        """Calculate the set of differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row):
        """Check if a row is strictly increasing and starts at 0."""
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True

    def generate_random_row(max_val, k):
        """
        Generate a random valid row of k+1 elements starting at 0, 
        with max element <= max_val.
        """
        if max_val < k:
            return None
        # We need to choose k numbers from 1..max_val such that they are distinct.
        # Then sort them.
        # But we also need to ensure the differences are unique within the row? 
        # No, the task says "all differences... are distinct over all rows".
        # Within a single row, differences must be distinct? 
        # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
        # This implies differences are unique across the ENTIRE set. 
        # Therefore, within a single row, differences must also be distinct.
        # A set of numbers where all pairwise differences are distinct is a Sidon set (B_2 set).
        
        # So we need to generate a Sidon set of size k+1 containing 0.
        
        # Simple random generation:
        # Start with [0]
        # Pick next element > current max.
        # Ensure new diffs don't conflict with existing diffs in the row.
        
        row = [0]
        used_diffs = set()
        current_max = 0
        
        for _ in range(k):
            # Try to pick a number
            # Range: current_max + 1 to max_val
            # We want small numbers to keep scope low.
            # Let's try picking from a small range first.
            found = False
            # To speed up, we can try a few random candidates
            # or a greedy scan from the bottom up.
            # Greedy scan is faster for finding *a* valid set, but might get stuck.
            # Random with retries is better for diversity.
            
            for _attempt in range(50):
                # Bias towards smaller numbers
                # max_val might be large, but we want small scope.
                # Let's pick a candidate between current_max + 1 and max_val
                # But we want to minimize the final max.
                # So we should try small candidates first.
                
                # Let's just iterate from current_max + 1 upwards
                # This is greedy.
                pass
                
            # Let's use a proper random Sidon generator
            # Pick x in [current_max + 1, max_val]
            # Check if {x - a for a in row} intersects used_diffs
            # If yes, reject.
            
            # To make it efficient, we'll try a limited number of random picks
            # biased towards the lower end.
            
            # Actually, for the candidate pool, we want *many* valid rows.
            # A simple recursive backtracking generator for Sidon sets is best.
            
            return None # Placeholder, will implement recursive gen below

    def generate_sidon_set(max_val, size, current_set, used_diffs, candidates):
        """
        Recursively generate Sidon sets of given size.
        """
        if len(candidates) > 1000: # Cap candidates to avoid memory blowup
            return
        
        if len(current_set) == size:
            candidates.append(tuple(current_set))
            return
            
        if not current_set:
            current_set = [0]
            # used_diffs is empty initially for the set itself, 
            # but we need to track diffs within this set.
            # Wait, the function signature is messy. 
            # Let's rewrite the generator logic in the main solve loop for clarity.
            pass

    def solve():
        # Strategy:
        # 1. Determine a lower bound for the max element.
        #    Counting bound: m >= ceil((n_rows * k*(k+1)/2) / (max_val)) ?
        #    Total differences = n_rows * k*(k+1)/2.
        #    Max difference is max_val.
        #    So max_val >= n_rows * k*(k+1)/2.
        #    For n=7, k=5: 7 * 15 = 105.
        
        total_diffs_needed = n_rows * (k * (k + 1) // 2)
        lower_bound = total_diffs_needed
        
        # We will search for a solution with max element starting from lower_bound upwards.
        
        # To generate candidates efficiently, we need a good Sidon set generator.
        # We will generate candidates for a specific max_val limit.
        
        def generate_candidates_for_limit(limit):
            """
            Generate a pool of valid Sidon rows with max element <= limit.
            """
            candidates = []
            
            # Use backtracking to generate Sidon sets of size k+1 starting with 0.
            # We want to generate many diverse ones.
            
            def backtrack(current_row, used_diffs):
                if len(current_row) == k + 1:
                    candidates.append(tuple(current_row))
                    return
                
                if len(candidates) >= 500: # Limit pool size
                    return
                    
                last_val = current_row[-1]
                
                # Next value must be > last_val and <= limit
                # Also, to keep the set "tight", we should prefer small values.
                # But we need to explore.
                
                # Heuristic: Try values from last_val + 1 to limit.
                # To avoid deep recursion on bad branches, we can prune.
                
                # Optimization: 
                # The minimum possible next value is last_val + 1.
                # The maximum possible next value is limit.
                
                # We will iterate through possible next values.
                # To get diverse candidates, we might want to randomize the order 
                # or just take the first N found.
                
                # Let's just find the first 500 valid sets by scanning from small to large.
                # This will bias towards "greedy" tight sets.
                
                for next_val in range(last_val + 1, limit + 1):
                    new_diffs = {next_val - a for a in current_row}
                    
                    # Check if new_diffs conflicts with used_diffs
                    if new_diffs & used_diffs:
                        continue
                    
                    # Also check if new_diffs has internal conflicts? 
                    # No, used_diffs contains all diffs from current_row.
                    # new_diffs are diffs between next_val and existing elements.
                    # Are there diffs between new elements? No, only one new element.
                    
                    # So we just check intersection.
                    
                    new_used_diffs = used_diffs | new_diffs
                    new_row = current_row + [next_val]
                    
                    backtrack(new_row, new_used_diffs)
                    
                    # Pruning: If we found enough, stop.
                    if len(candidates) >= 500:
                        return

            backtrack([0], set())
            return candidates

        # Main search loop
        current_limit = lower_bound
        
        while time.time() - start_time < time_limit:
            # Generate candidates for this limit
            # Note: Generating 500 candidates via backtracking might be slow if limit is large.
            # But limit starts at 105.
            
            candidates = generate_candidates_for_limit(current_limit)
            
            if not candidates:
                current_limit += 5
                continue
                
            # Now we have a pool of candidate rows.
            # We need to select n_rows disjoint rows from this pool.
            # This is a Set Packing problem.
            # Since n_rows is small, we can use DFS with pruning.
            
            # Precompute difference sets for all candidates
            cand_diffs = [get_diffs(list(c)) for c in candidates]
            cand_maxs = [max(c) for c in candidates]
            
            # Sort candidates by max value (ascending) to prefer smaller scopes
            # We create a list of (max_val, index)
            sorted_indices = sorted(range(len(candidates)), key=lambda i: cand_maxs[i])
            
            # We want to find a subset of size n_rows such that no two share a difference.
            
            def dfs(selected, used_diffs_global, next_start_idx):
                if len(selected) == n_rows:
                    return selected
                
                if next_start_idx >= len(sorted_indices):
                    return None
                    
                # Pruning: 
                # If the current max scope is already >= best known, we could prune?
                # We are searching for *any* valid set, then we can try to minimize.
                # But since we iterate limits upwards, the first valid set we find 
                # at this limit is the best for this limit.
                
                for i in range(next_start_idx, len(sorted_indices)):
                    idx = sorted_indices[i]
                    row = candidates[idx]
                    diffs = cand_diffs[idx]
                    
                    # Check conflict
                    if diffs & used_diffs_global:
                        continue
                    
                    # Check if adding this row could possibly lead to a solution?
                    # Not easy to prune without more info.
                    
                    # Recurse
                    result = dfs(selected + [row], used_diffs_global | diffs, i + 1)
                    if result is not None:
                        return result
                    
                return None

            # Run DFS
            # To speed up, we can limit the search depth or breadth?
            # The pool is 500. n_rows is 7.
            # Branching factor is ~500. Depth 7. 500^7 is too big.
            # But we prune heavily with conflicts.
            # Also, we sorted by max value, so we try small ones first.
            
            # To make it faster, we can restrict the pool to the top K candidates 
            # with smallest max values.
            # Let's try with the full pool first, but with a time check.
            
            # Optimization: 
            # We can use a greedy approach first.
            # Pick the smallest candidate.
            # Remove all conflicting candidates.
            # Repeat.
            # If we get n_rows, we have a solution.
            
            greedy_selected = []
            remaining = set(range(len(candidates)))
            used_diffs_greedy = set()
            
            # Sort candidates by max value
            for idx in sorted_indices:
                if len(greedy_selected) == n_rows:
                    break
                if idx not in remaining:
                    continue
                
                diffs = cand_diffs[idx]
                if diffs & used_diffs_greedy:
                    continue
                    
                # Greedy pick
                greedy_selected.append(candidates[idx])
                used_diffs_greedy |= diffs
                
                # Remove conflicting candidates from remaining
                # We can just mark them as unavailable in the next step
                # For efficiency, we just check against used_diffs_greedy in the loop above.
                # But we also need to ensure we don't pick the same candidate twice (handled by unique indices).
                
            if len(greedy_selected) == n_rows:
                # We found a valid set with max scope <= current_limit
                # Is it optimal? 
                # Since we iterated limits upwards, yes, this is the minimum limit 
                # for which a greedy solution exists.
                # However, a greedy solution might not exist even if a non-greedy one does.
                # If greedy fails, we should try DFS.
                
                # Let's return the greedy solution if found.
                # It minimizes the max element within the greedy heuristic.
                return [list(row) for row in greedy_selected]
            else:
                # Greedy failed. Try DFS.
                # To avoid timeout, we limit DFS iterations or pool size.
                # Let's try DFS on a smaller pool: top 100 candidates.
                top_indices = sorted_indices[:100]
                
                def dfs_limited(selected, used_diffs_global, next_start_idx, pool):
                    if len(selected) == n_rows:
                        return selected
                    if next_start_idx >= len(pool):
                        return None
                    for i in range(next_start_idx, len(pool)):
                        idx = pool[i]
                        diffs = cand_diffs[idx]
                        if diffs & used_diffs_global:
                            continue
                        result = dfs_limited(selected + [candidates[idx]], used_diffs_global | diffs, i + 1, pool)
                        if result is not None:
                            return result
                    return None

                solution = dfs_limited([], set(), 0, top_indices)
                if solution:
                    return [list(row) for row in solution]
            
            # If no solution found at this limit, increase limit
            current_limit += 5
            
        # If time runs out, return the best greedy attempt or empty?
        # The task requires a valid result.
        # If we haven't found one, we should fall back to a valid construction.
        # Greedy init from the original code is a good fallback.
        
        def greedy_init():
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            break
                        x += 1
                    row.append(x)
                    used_diffs.update({x - a for a in row[:-1]})
                rows.append(row)
            return rows

        return greedy_init()

    return solve()
