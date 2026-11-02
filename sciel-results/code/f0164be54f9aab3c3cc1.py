import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    # The evaluator kills at 150s. We use 140s as a hard stop to be safe,
    # but aim to finish earlier if possible.
    deadline = start_time + 140

    # Total number of differences required
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on scope M:
    # We need total_diffs distinct integers in [1, M-1].
    # So M-1 >= total_diffs => M >= total_diffs + 1
    # Also each row has k+1 elements, max element M_r.
    # Simple bound: M >= total_diffs + 1
    min_scope = total_diffs + 1

    # A slightly tighter bound:
    # For a single row of length k+1 with max M_r, the number of diffs is k(k+1)/2.
    # The diffs must be distinct.
    # If we have n_rows rows, total diffs is n_rows * k(k+1)/2.
    # All these diffs must be in [1, M_max - 1].
    # So M_max >= total_diffs + 1.

    # Let's also consider the structure.
    # For k=7, k(k+1)/2 = 28 diffs per row.
    # If n_rows=4, total diffs = 112.
    # So M >= 113.
    # The problem statement says simple counting bound is 112. This likely refers to M-1 >= 112 => M >= 113.
    # Best known is 134.

    def get_diffs(row):
        """Calculate the set of positive differences for a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def is_valid(rows):
        """Check if the set of rows is valid (all diffs distinct across all rows)."""
        used = set()
        for row in rows:
            d = get_diffs(row)
            if d & used:
                return False
            used |= d
        return True

    def greedy_construct(limit, seed=None):
        """
        Try to construct n_rows rows with max entry <= limit.
        Uses a backtracking greedy approach.
        """
        if seed is not None:
            random.seed(seed)

        used = set()
        rows = []

        for r_idx in range(n_rows):
            row = [0]
            # We need to add k more elements
            # Backtracking search
            def backtrack(row, used_local, target_limit):
                if len(row) == k + 1:
                    return True
                
                last = row[-1]
                # Try next elements
                # Optimization: We can jump to a smaller range if we are close to limit
                # But we must ensure we can fit the remaining elements.
                # Min next value: last + 1
                # Max next value: target_limit
                
                # To speed up, we can randomize the order of trying candidates slightly?
                # Or just iterate.
                
                # Heuristic: Try smaller values first to keep the row compact?
                # Or larger values to spread out?
                # Usually, for packing, spreading out might be better to avoid conflicts?
                # But we want to minimize the MAX entry, so we want to stay small?
                # Actually, if we stay too small, we might run out of space for diffs.
                # Let's try a mix.
                
                # Generate candidates
                # We can limit the search space.
                # If we pick x, the new diffs are {x - a for a in row}.
                
                # Let's iterate x from last + 1 to target_limit
                # But this can be slow if target_limit is large.
                # However, k is small (7).
                
                # A better heuristic:
                # The diffs must be distinct.
                # If we pick x, we use diffs x - row[0], x - row[1], ...
                
                # Let's just iterate.
                # To add randomness, we can shuffle the range? No, x is ordered.
                # We can skip some x with a probability?
                
                # Let's try a "randomized greedy" where we pick a random valid x from a window?
                # No, backtracking is better for finding *any* solution.
                
                # To make it faster, we can prune.
                # If we pick x, we must be able to complete the row.
                
                # Simple iteration
                # We can try to bias towards the middle or end?
                # Let's just iterate from last+1 to target_limit.
                
                # Optimization: If we have many candidates, we might want to try some at random?
                # For now, standard DFS.
                
                # To avoid deep recursion limits or slow performance, we can use a stack?
                # Recursion depth is k, which is small.
                
                # One issue: If we always pick the smallest valid x, we might get stuck.
                # We should randomize the choice among valid candidates.
                
                valid_candidates = []
                for x in range(last + 1, target_limit + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_local):
                        valid_candidates.append(x)
                    # Early stop if we have enough candidates?
                    # No, we want to explore.
                    # But if target_limit is large, this loop is long.
                    
                    # Heuristic: If we find many candidates, we can break and pick one?
                    # This makes it greedy-randomized rather than full backtracking.
                    # Let's do a hybrid: collect up to 20 candidates, then pick one randomly.
                    # If we fail, we backtrack.
                    if len(valid_candidates) >= 20:
                        break

                if not valid_candidates:
                    return False

                # Pick a candidate
                # With some probability, pick the first (greedy)
                # Otherwise, pick a random one
                if random.random() < 0.3:
                    x = valid_candidates[0]
                else:
                    x = random.choice(valid_candidates)

                new_diffs = {x - a for a in row}
                row.append(x)
                used_local |= new_diffs

                if backtrack(row, used_local, target_limit):
                    return True

                # Backtrack
                row.pop()
                used_local -= new_diffs

                # Try next candidate?
                # In this simplified version, we only try one path per call level?
                # No, we need to loop over candidates in the parent call.
                # The current structure only tries ONE candidate per level (the chosen one).
                # If it fails, it returns False.
                # This is not full backtracking. It's a single randomized path.
                # To do proper backtracking, we need to iterate through ALL valid_candidates.
                
                # Let's rewrite backtrack to iterate.
                return False

            # Proper Backtracking
            def backtrack_full(row, used_local, target_limit):
                if len(row) == k + 1:
                    return True
                
                last = row[-1]
                
                # We need to try all valid x
                # To speed up, we can limit the number of candidates we consider at each step?
                # Or just iterate all.
                
                # Iterate x
                # To add randomness, we can shuffle the order of x?
                # x is from last+1 to target_limit.
                # Shuffling a large range is expensive.
                
                # Let's stick to iterating, but maybe skip some?
                
                # Optimization:
                # If we are at depth d, we have k-d elements left.
                # The minimum possible max value for the rest of the row is last + (k-d) * 1 = last + k - d.
                # So if last + k - d > target_limit, we can prune.
                
                if last + (k - (len(row) - 1)) > target_limit:
                    return False

                # Generate candidates
                # We can collect all valid x, then shuffle them to try in random order.
                valid_candidates = []
                for x in range(last + 1, target_limit + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_local):
                        valid_candidates.append(x)
                
                if not valid_candidates:
                    return False

                # Shuffle candidates to avoid deterministic failures
                random.shuffle(valid_candidates)
                
                # To limit search space, we can only try the first N candidates?
                # This makes it a "beam search" of width 1 with randomization?
                # No, if we only try the first one, we are not backtracking fully.
                # Let's try all, but with a time check?
                
                # Given the small k, trying all might be okay.
                # But target_limit can be up to 134.
                # Range size ~ 100.
                # Depth 7.
                # 100^7 is huge.
                
                # We MUST prune heavily.
                # The "collect up to 20 and pick one" strategy is effectively a randomized greedy with limited backtracking.
                # Let's implement a limited backtracking: try up to MAX_BACKTRACK candidates per level.
                
                MAX_CANDIDATES = 10
                if len(valid_candidates) > MAX_CANDIDATES:
                    # Pick a random subset
                    valid_candidates = random.sample(valid_candidates, MAX_CANDIDATES)
                
                for x in valid_candidates:
                    new_diffs = {x - a for a in row}
                    row.append(x)
                    used_local |= new_diffs
                    
                    if backtrack_full(row, used_local, target_limit):
                        return True
                    
                    row.pop()
                    used_local -= new_diffs

                return False

            # Start backtracking
            if not backtrack_full(row, used, limit):
                return None

            rows.append(row)
            used |= get_diffs(row)

        return rows

    # Main Search Loop
    # Start with a generous limit and decrease
    # Or start with the lower bound and increase?
    # Decreasing is better for minimization.
    # Start with a known good bound or an estimate.
    # Estimate: total_diffs + n_rows * k?
    
    # Let's start with M = total_diffs + 10
    current_limit = min_scope + 20
    
    best_rows = None
    best_scope = float('inf')

    # Try to find a solution with current_limit
    # If found, try to lower it.
    
    # We can use a binary search or linear decrease.
    # Linear decrease is safer.
    
    # To speed up, we can try multiple seeds for each limit.
    
    attempts_per_limit = 5
    
    # We will iterate limits downwards
    # But we don't know the exact minimum.
    # Let's start high and go down.
    
    # Initial guess: 150
    current_limit = 150
    
    # If we can't find a solution at 150, increase?
    # No, 134 is known to be possible.
    # So 150 should be easy.
    
    # Let's find the minimum M.
    
    # Strategy:
    # 1. Find any valid solution with M=150.
    # 2. Try M=149, 148, ...
    # 3. Stop when we fail for M-1, or time runs out.
    
    # But we want to MINIMIZE the scope.
    # So we want the smallest M for which a solution exists.
    
    # Let's do a binary search on M?
    # [min_scope, 150]
    
    low = min_scope
    high = 150
    
    best_solution = None
    
    # We want to find the smallest M.
    # So we check if M is feasible.
    
    # Feasibility check:
    def is_feasible(M):
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline:
                return None
            rows = greedy_construct(M, seed=seed)
            if rows is not None:
                return rows
        return None

    # Binary search for the minimum M
    # But the feasibility is non-deterministic due to randomization.
    # So we might miss a feasible M.
    # Better to scan downwards from a known feasible point?
    # Or scan upwards from the lower bound?
    
    # Scanning upwards from min_scope is safer to ensure we find the MINIMUM.
    # But if the minimum is high, it takes long.
    # Scanning downwards from 150 is faster to find a good solution, but might miss the optimal if we stop too early.
    
    # Let's combine:
    # 1. Find a solution with M=150.
    # 2. Try to lower M one by one.
    # 3. If we fail for M, try M+1 again with more seeds?
    
    # Actually, let's just scan from min_scope upwards.
    # min_scope is 113.
    # 113, 114, ...
    
    # If we find a solution at M, we can stop?
    # No, we want the MINIMUM.
    # So if we find a solution at M, we should check if M-1 is feasible.
    # If M-1 is not feasible, then M is the minimum.
    
    # So:
    # 1. Find smallest M >= min_scope such that is_feasible(M) is True.
    
    # To speed up, we can parallelize? No.
    # We can increase the number of attempts for smaller M?
    
    # Let's try scanning from min_scope.
    
    current_M = min_scope
    
    while time.time() < deadline:
        if current_M > 150: # Safety break
            break
            
        # Try to find a solution for current_M
        found = None
        # Increase attempts as M gets closer to the likely minimum?
        # Or just fixed attempts.
        attempts = 10 if current_M < 120 else 5
        
        for seed in range(attempts):
            if time.time() > deadline:
                break
            rows = greedy_construct(current_M, seed=seed)
            if rows is not None:
                found = rows
                break
        
        if found is not None:
            # We found a solution for current_M
            # Check if current_M - 1 is feasible
            if current_M > min_scope:
                # Try to see if we can go lower
                # This is tricky because we already spent time on current_M.
                # We should have checked current_M - 1 first?
                # No, we are scanning upwards.
                # If we found a solution at M, the minimum is <= M.
                # It could be M, M-1, ...
                # Since we scanned from min_scope upwards, if we found a solution at M,
                # and we didn't find one at M-1 (because we would have stopped earlier?),
                # Wait, if we found a solution at M-1, we would have recorded it and then tried M-2?
                # No, the logic "if found, check M-1" implies we found M, then check M-1.
                # If M-1 is feasible, we should update best to M-1 and continue checking M-2.
                
                # Let's restructure:
                # We want the MINIMUM M.
                # So we should check M, M-1, M-2 ... until we fail.
                
                # But we don't know where to start.
                # Let's start from min_scope and go UP.
                # If we find a solution at M, we know the minimum is <= M.
                # But we might have missed a smaller one?
                # No, we are going up from min_scope.
                # So if we find a solution at M, and we didn't find one at M-1, M-2, etc.,
                # then M is the minimum.
                
                # So the loop should be:
                # for M in range(min_scope, 151):
                #    if is_feasible(M):
                #       best = M
                #       break
                #    else:
                #       continue
                
                # This finds the FIRST feasible M.
                # This is the minimum feasible M.
                
                # So if we found a solution at current_M, we can return it?
                # Yes, because we are scanning from the bottom.
                
                # However, due to randomness, we might FAIL to find a solution at M-1 even if it exists.
                # So we might return M when M-1 is actually feasible.
                # To mitigate this, we can retry M-1 with more seeds if we find M?
                # Or just accept the risk.
                
                # Let's add a check: if we found a solution at M, try M-1 with 20 seeds.
                # If M-1 is feasible, update M to M-1 and try M-2?
                
                # This could be expensive.
                # Let's just return the first found solution.
                # It's a good heuristic.
                
                # Update best
                best_rows = found
                best_scope = current_M
                break
            else:
                # No solution found for current_M
                current_M += 1
        else:
            # Exhausted attempts for current_M
            current_M += 1

    # If we didn't find anything in the loop (e.g. time ran out), try a fallback
    if best_rows is None:
        # Try a large M
        for M in [140, 150, 160]:
            if time.time() > deadline:
                break
            rows = greedy_construct(M, seed=1)
            if rows is not None:
                best_rows = rows
                best_scope = M
                break

    if best_rows is None:
        # Last resort: return a valid but potentially large set
        # Construct using greedy without limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = max(max(r) for r in best_rows)

    return best_rows
