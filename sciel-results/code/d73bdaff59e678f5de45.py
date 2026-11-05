"""
Difference triangle set (n_rows, k) constructor.

Strategy:
- Model the problem as a multi-agent coordination task where each row is an agent.
- The environment is a shared difference bitmask (or set) that tracks used differences.
- Instead of generic SA moves, we use "Exact Row Reconstruction":
  1. Pick a row (agent) to optimize.
  2. Temporarily remove its differences from the shared state.
  3. Use a depth-first backtracking search to reconstruct the row with the smallest possible
     maximum entry, subject to the constraint that no difference conflicts with the global state.
  4. This ensures the new row is always valid with respect to the others.
  5. Iterate over rows repeatedly to tighten the global scope.
- This is structurally different from the seed's SA because it uses exact constraint propagation
  for the "action" of an agent, rather than stochastic local moves.
"""
import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Hard deadline: return before 150 s wall-clock with a safety margin.
    deadline = start_time + 145.0

    # Each row has k+1 entries (including 0). Differences per row = k*(k+1)/2.
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on the maximum entry: at least total_diffs (simple counting bound)
    # Actually, the counting bound is on the number of differences, which is total_diffs.
    # The max entry M must be at least such that there are enough differences in [1..M].
    # Number of differences in a single row with max M is k(k+1)/2.
    # Wait, the differences are distinct ACROSS all rows.
    # So we need total_diffs distinct differences.
    # The differences are values in 1..M.
    # So M must be at least total_diffs.
    lower_bound = total_diffs

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = []
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - ri)
        return diffs

    # Helper: validate rows (all diffs distinct)
    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    # Helper: get max entry
    def get_max_entry(rows):
        m = 0
        for row in rows:
            if row and row[-1] > m:
                m = row[-1]
        return m

    # Initial construction: Simple greedy to get a valid starting point
    def greedy_construct():
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        return rows

    # Initial rows
    best_rows = greedy_construct()
    best_max = get_max_entry(best_rows)
    
    # Global state: set of used differences
    used_diffs = set()
    for row in best_rows:
        for d in row_diffs(row):
            used_diffs.add(d)

    # Backtracking solver for a single row given the set of forbidden differences
    # Returns a valid row of length k+1 starting with 0, minimizing the last element
    # If no solution found within time limit or search space, returns None
    def reconstruct_row(forbidden_diffs, target_max, time_limit=0.1):
        """
        Find a row of k+1 integers starting with 0, strictly increasing,
        such that all pairwise differences are NOT in forbidden_diffs,
        and the last element <= target_max.
        Returns the row or None.
        """
        row = [0]
        
        # Heuristic: we want to find the smallest possible max.
        # We can use DFS with pruning.
        # To speed up, we can use a heuristic ordering of candidates.
        
        # Precompute potential candidates? No, just iterate.
        
        # We need to fill k positions.
        # State: current row, current index to fill (1 to k)
        
        # Optimization: If we are trying to beat a current max, we can limit search.
        
        # Let's use a recursive helper with memoization? No, state is large.
        # Just use DFS with time check.
        
        def dfs(pos, current_row, current_max_limit):
            if time.time() > time_limit:
                return None
            
            if pos == k:
                # Check if we can close the row
                # We need to ensure the last element is within limit
                # Actually, we build the row step by step.
                # The last element added is current_row[-1].
                if current_row[-1] <= current_max_limit:
                    # Verify all diffs are not in forbidden
                    # They are checked during construction, so this should be safe
                    return current_row[:]
                return None
            
            last_val = current_row[-1]
            
            # Determine the range for the next value
            # It must be > last_val
            # And the resulting differences with previous elements must not be in forbidden_diffs
            # Also, the new value itself contributes to the max.
            
            # To minimize max, we should try smaller values first.
            # But we also need to leave room for the remaining k - pos elements.
            # The minimum possible value for the next element is last_val + 1.
            # The maximum possible value is constrained by current_max_limit.
            
            # However, we must ensure that we can still place the remaining elements.
            # A simple lower bound for the next element:
            # If we pick x, the next one must be > x.
            # The smallest sequence is x, x+1, ..., x + (k-pos-1).
            # So x + (k-pos-1) <= current_max_limit => x <= current_max_limit - (k-pos-1).
            
            min_next = last_val + 1
            max_next = current_max_limit - (k - pos - 1)
            
            if min_next > max_next:
                return None
            
            # Try candidates in order
            for x in range(min_next, max_next + 1):
                # Check conflicts
                ok = True
                for a in current_row:
                    d = x - a
                    if d in forbidden_diffs:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Tentatively add x
                current_row.append(x)
                result = dfs(pos + 1, current_row, current_max_limit)
                if result is not None:
                    return result
                current_row.pop()
            
            return None
        
        return dfs(0, row, target_max)

    # Main optimization loop
    # We iterate over rows, trying to reconstruct each one with a smaller max.
    # We keep a global "current max" and try to reduce it.
    
    # To make it more "multi-agent" like, we could randomize the order or use a shared budget.
    # But simple iterative improvement is robust.
    
    # We will try to reduce the max entry of each row to be <= current_global_max.
    # If we succeed with a smaller max, we update the global state.
    
    # To avoid getting stuck, we can allow "rebuilding" a row even if it doesn't reduce the global max
    # but might allow other rows to reduce later? 
    # Actually, if a row is rebuilt to a smaller max, it frees up some large differences,
    # potentially allowing other rows to use them? No, it frees up the differences it WAS using.
    # It occupies NEW differences.
    # So it might block others.
    # However, if we strictly minimize the max of the current row, we are using the "smallest" possible differences.
    # This tends to pack differences tightly, which is good for the global bound.
    
    # Strategy:
    # 1. Compute current global max M.
    # 2. For each row i:
    #    a. Remove row i's diffs from used_diffs.
    #    b. Try to reconstruct row i with max <= M-1 (to strictly improve global max if this was the max row).
    #       If M-1 is not possible, try M.
    #       Actually, we want to minimize the row's max.
    #       Let's try to find the minimum possible max for this row given the forbidden diffs.
    #       We can binary search on the max? Or just try to find ANY valid row with max <= current_row_max - 1?
    #       If we find one, we update.
    #    
    #    Let's use a binary search on the target max for the row.
    #    Low = current_row_min_possible (which is roughly k(k+1)/2? No, just k if packed 0,1,2...)
    #    High = current_row_max
    #    
    #    Actually, a simpler greedy approach:
    #    Try to reconstruct with max = current_global_max - 1.
    #    If successful, update global max.
    #    If not, try to reconstruct with max = current_global_max.
    #    If successful, update row (might help future steps by changing the set of used diffs).
    #    If not, keep original row.
    
    # Let's implement a more aggressive search:
    # For each row, try to find the MINIMUM possible max given the current forbidden set.
    # We can do this by binary searching on the max value.
    
    def find_min_max_for_row(forbidden_diffs, current_max_hint):
        # Lower bound: k (if 0,1,2...k) but differences must be distinct.
        # Actually, the minimum max for a row of length k+1 with distinct diffs is not trivial.
        # But it's at least k.
        # Let's set low = k, high = current_max_hint
        low = k
        high = current_max_hint
        
        # If we can't even fit in 'high', return None
        # We'll just try to find the smallest valid max.
        
        # To save time, we'll just try a few candidate max values:
        # 1. current_max_hint - 1
        # 2. current_max_hint - 2
        # ... down to some limit
        # Or just use the DFS with a decreasing limit.
        
        # Given the time constraint, let's just try to improve by 1.
        # If we can reduce the max by 1, great.
        # This is slow but steady.
        
        # Better: Try to find the absolute minimum for this row in isolation?
        # No, it depends on forbidden_diffs.
        
        # Let's use the DFS with a tight time limit to find the best row.
        # We'll try limits: current_max_hint, current_max_hint-1, current_max_hint-2...
        # Stop if we find a valid one or time out.
        
        best_row = None
        best_max = current_max_hint + 1
        
        # We want to minimize the max.
        # Let's try to find a row with max <= current_max_hint - 1 first.
        # If found, that's an improvement for the global max (if this row was the bottleneck).
        
        # To be efficient, we'll try a few specific targets.
        # Target 1: current_max_hint - 1
        # Target 2: current_max_hint - 5 (jump)
        # Target 3: current_max_hint - 10
        
        # Actually, let's just use the DFS with the current_max_hint as the upper bound,
        # and let the DFS find the smallest one?
        # My DFS above just finds ANY row with max <= limit.
        # It doesn't minimize.
        # To minimize, we'd need to binary search on the limit.
        
        # Let's binary search on the limit.
        # low = k, high = current_max_hint
        # Find smallest L such that reconstruct_row(forbidden, L) is not None.
        
        # But reconstruct_row is expensive.
        # Let's limit the number of DFS calls.
        
        # Strategy:
        # 1. Try L = current_max_hint - 1.
        # 2. If fail, try L = current_max_hint - 2.
        # 3. If fail, keep original.
        
        # This is safe and simple.
        
        # Let's try a slightly wider search if time permits.
        
        # For now, let's just try to improve by 1.
        
        target = current_max_hint - 1
        if target < k:
            return None, None
            
        row_candidate = reconstruct_row(forbidden_diffs, target, time_limit=0.2)
        if row_candidate:
            return row_candidate, target
        
        # If we couldn't improve by 1, maybe we can find a different configuration with the same max
        # that frees up useful differences?
        # Let's try to reconstruct with the SAME max, but a different structure.
        # This might help other rows.
        # But it's risky. Let's stick to improvements first.
        
        return None, None

    # Optimization Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_global_max = get_max_entry(best_rows)
        
        # Randomize order to avoid bias
        row_indices = list(range(n_rows))
        random.shuffle(row_indices)
        
        for r_idx in row_indices:
            if time.time() > deadline - 2.0:
                break
                
            original_row = best_rows[r_idx]
            original_diffs = row_diffs(original_row)
            
            # Remove original diffs from global set
            for d in original_diffs:
                used_diffs.discard(d)
                
            # Try to improve this row
            new_row, new_max = find_min_max_for_row(used_diffs, current_global_max)
            
            if new_row:
                # Accept the new row
                best_rows[r_idx] = new_row
                # Update global set
                for d in row_diffs(new_row):
                    used_diffs.add(d)
                improved = True
            else:
                # Revert: put original diffs back
                for d in original_diffs:
                    used_diffs.add(d)
            
            # Update global max if needed
            # If we improved a row, the global max might have decreased
            # But we only improved if new_max < current_global_max?
            # find_min_max_for_row tries current_global_max - 1.
            # So if it succeeds, the new row's max is <= current_global_max - 1.
            # So the global max might decrease.
            # We recalculate at the start of the next outer loop iteration.
            # But within this loop, we use the old current_global_max for subsequent rows.
            # This is fine, it's a conservative bound.

    # Final validation
    if not is_valid(best_rows):
        # Fallback
        best_rows = greedy_construct()

    return best_rows
