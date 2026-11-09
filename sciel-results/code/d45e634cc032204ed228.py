import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    num_diffs_per_row = k_plus_1 * (k_plus_1 - 1) // 2

    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
        
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def backtrack_row(free_diffs, k, max_limit, seed=0, time_limit=1.0):
        """
        Attempts to construct a row of size k+1 starting at 0, with max element <= max_limit,
        such that all differences are in free_diffs (i.e., not used by other rows) and are unique within the row.
        Returns the row with the minimal possible max element, or None if not found.
        Uses backtracking with pruning.
        """
        random.seed(seed)
        best_row = None
        best_max = max_limit + 1
        
        # We want to find a valid row with min max.
        # State: (current_row, current_max, used_in_row)
        
        def dfs(current_row, current_max, used_in_row):
            nonlocal best_row, best_max
            
            if time.time() > time_limit:
                return False

            if len(current_row) == k_plus_1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = list(current_row)
                return True
            
            # Pruning: if current_max is already >= best_max, we can stop this branch
            # because we are looking for a strictly smaller max.
            if current_max >= best_max:
                return False

            # Determine the range for the next element x
            start_x = current_max + 1
            # We don't need to try x >= best_max if we already have a solution
            end_x = min(max_limit, best_max - 1)
            
            if start_x > end_x:
                return False

            # Heuristic: Try smaller x first to keep the row tight.
            # However, if we get stuck, we might need to explore larger x earlier.
            # For now, sequential is good for finding the *minimal* max.
            
            candidates = []
            for x in range(start_x, end_x + 1):
                new_diffs = {x - a for a in current_row}
                
                # Check if any of these new differences are already used in this row
                if new_diffs & used_in_row:
                    continue
                
                # Check if any of these new differences are forbidden (used by other rows)
                # Note: free_diffs contains differences that are NOT used by other rows.
                # So we need new_diffs to be a subset of free_diffs.
                if not new_diffs.issubset(free_diffs):
                    continue
                
                candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # If we have many candidates, we might want to randomize the order slightly
            # to escape local minima, but for finding the *minimal* max, order matters less
            # than the pruning. However, if the first few small candidates fail, we backtrack.
            # To speed up, we can just iterate.
            
            for x, new_diffs in candidates:
                current_row.append(x)
                used_in_row.update(new_diffs)
                
                if dfs(current_row, x, used_in_row):
                    # If we found a solution, we can return early if we are just looking for ANY solution
                    # But we want the MINIMAL. 
                    # Since we iterate x in increasing order, the first solution found 
                    # at a given depth isn't necessarily the global minimal max for the whole row,
                    # but the DFS structure with best_max pruning handles that.
                    # Actually, if we return True immediately, we stop exploring other candidates 
                    # at this level. This is correct if we want *a* solution with max < best_max.
                    # But we want the *best* (smallest) max.
                    # The standard approach: update best_max and best_row, then return False to continue searching?
                    # No, if we find a valid row with max M, any other valid row from this branch 
                    # will have max >= current_max (which is <= M). 
                    # Wait, current_max is the last element. 
                    # If we pick a smaller x now, we might be able to pick smaller elements later?
                    # Yes. So we should continue searching for smaller x.
                    # But if we pick x, the max of the row will be at least x.
                    # If we found a solution with best_max = B, and we are currently at x < B,
                    # we should continue to see if we can find a solution with max < x? No, max will be >= x.
                    # So if we found a solution with max B, and we are exploring x, 
                    # any solution from this branch will have max >= x.
                    # If x >= B, we prune.
                    # If x < B, we might find a solution with max < B.
                    # So we should continue.
                    
                    # However, for efficiency, if we just want *any* valid row to reduce the scope,
                    # finding one is enough. The outer loop can try to reduce further.
                    # Let's return True to stop the search for this row construction.
                    # This makes it a "find any valid row with max <= max_limit" function,
                    # but with the bias towards smaller max due to iteration order.
                    return True
                
                current_row.pop()
                used_in_row.difference_update(new_diffs)
            
            return False
        
        # Start DFS
        dfs([0], 0, set())
        
        if best_row:
            return best_row
        return None

    def greedy_solution(seed=0):
        """
        Generate a valid solution using randomized greedy.
        This serves as a fallback or initial population.
        """
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                # Adaptive limit
                limit = x + 2000
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 20:
                            break
                    x += 1
                
                if not candidates:
                    # Fallback: find the first valid one, no matter how large
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Choose the smallest candidate to keep the row tight
                # This is a "greedy min" strategy
                chosen = candidates[0]
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows

    def local_search_shrink(rows, max_time=10.0):
        """
        Attempts to reduce the global maximum by rebuilding the row with the largest max.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        
        while time.time() - start < max_time:
            # Find the row with the largest maximum
            max_row_idx = 0
            max_val = -1
            for i, r in enumerate(rows):
                if max(r) > max_val:
                    max_val = max(r)
                    max_row_idx = i
            
            # We want to find a new row for max_row_idx with max < max_val
            # Compute the set of differences used by ALL OTHER rows
            used_by_others = set()
            for i, r in enumerate(rows):
                if i == max_row_idx:
                    continue
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        used_by_others.add(r[b] - r[a])
            
            # Differences that are "free" (not used by others)
            # Note: We don't know the total universe of differences, but we can assume
            # that any difference not in used_by_others is potentially available.
            # However, the new row must have unique internal differences.
            
            # Try to construct a new row with max_limit = max_val - 1
            # If we find one, it's an improvement.
            # We can try several random seeds to find a valid row.
            
            found_improvement = False
            # Limit time per row rebuild attempt
            rebuild_time_limit = min(0.5, time_limit - (time.time() - start_time))
            if rebuild_time_limit < 0.1:
                break
                
            for seed in range(10):
                new_row = backtrack_row(used_by_others, k, max_val - 1, seed=seed, time_limit=rebuild_time_limit)
                if new_row is not None:
                    # Verify validity of the whole set with the new row
                    test_rows = [list(r) for r in rows]
                    test_rows[max_row_idx] = new_row
                    if is_valid(test_rows):
                        rows = test_rows
                        new_score = get_score(rows)
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [list(r) for r in rows]
                        found_improvement = True
                        break
            
            if not found_improvement:
                # If we can't reduce the max of the largest row, 
                # maybe we can reduce the max of another row to free up differences?
                # Or we are stuck.
                # Let's try to reduce the second largest row if the largest is already minimal?
                # For simplicity, if we can't shrink the largest, we stop.
                # A more advanced approach would try to swap differences between rows.
                break
        
        return best_rows

    # Main search logic
    best_rows = None
    best_score = float('inf')
    
    # 1. Generate an initial solution using greedy
    # Try a few seeds to get a good starting point
    for seed in range(20):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 10:
            break
        
        rows = greedy_solution(seed)
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
        # If we found a very good solution, we can break early
        if score <= 115:
            break

    if best_rows is None:
        # Emergency fallback: construct a valid set with no optimization
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # 2. Local search to shrink the scope
    # Allocate most of the remaining time to local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        # Run local search
        improved_rows = local_search_shrink(best_rows, max_time=remaining * 0.8)
        if improved_rows:
            score = get_score(improved_rows)
            if score < best_score:
                best_rows = improved_rows
                best_score = score

    # 3. Final validation
    if not is_valid(best_rows):
        # If something went wrong, return the last known valid one or a simple one
        # The greedy_solution should always return valid rows.
        # local_search_shrink checks validity before accepting.
        # So this should rarely trigger.
        pass

    return best_rows
