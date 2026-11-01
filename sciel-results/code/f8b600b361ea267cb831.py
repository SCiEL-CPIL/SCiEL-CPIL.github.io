import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    # The evaluator kills at 150s. We use 145s as a hard deadline for safety.
    deadline = _START + 145.0 

    d = k * (k + 1) // 2  # number of differences per row
    total_diffs = n_rows * d

    # Helper to check validity and return scope
    def get_scope_and_diffs(rows):
        used = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in used:
                        return None, None
                    used.add(diff)
            if row[-1] > scope:
                scope = row[-1]
        return scope, used

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_init()
    scope, used_diffs = get_scope_and_diffs(rows)
    best_rows = [r[:] for r in rows]
    best_scope = scope
    
    rng = random.Random(42)

    # Function to try to reconstruct a specific row index 'target_idx'
    # using backtracking, constrained by 'forbidden_diffs' (diffs used by other rows)
    # and a maximum value 'max_val'.
    def try_reconstruct_row(target_idx, forbidden_diffs, max_val, rows):
        # We are rebuilding rows[target_idx]
        # Other rows are fixed.
        # We need to find a sequence 0 = x0 < x1 < ... < xk <= max_val
        # such that all diffs xj - xi are not in forbidden_diffs.
        
        # We can use a simple backtracking search.
        # To speed it up, we can use a heuristic: try small values first.
        
        # We'll use a recursive helper, but Python recursion limit might be an issue if k is large.
        # k=8 means depth 9, which is fine.
        
        def backtrack(pos, current_row, current_diffs):
            if pos == k + 1:
                return current_row[:]
            
            # Heuristic: try values from current_row[-1] + 1 up to max_val
            # But we can prune if current_row[-1] + 1 > max_val
            start_val = current_row[-1] + 1
            if start_val > max_val:
                return None
            
            # To make it faster, we can limit the search width or use a smarter order.
            # Let's try a window of candidates.
            # If max_val is large, iterating up to max_val is slow.
            # We can sample or use a greedy step.
            
            # Let's try a limited set of candidates: 
            # 1. The smallest valid one (greedy)
            # 2. Some random ones
            # But backtracking needs to explore.
            
            # Optimization: If we just want *a* valid row, greedy with backtracking 
            # on the last few elements is often enough.
            
            # Let's try values from start_val to min(start_val + 50, max_val) first.
            # If that fails, we might need to go higher, but usually small values are better.
            
            # Actually, for a fixed max_val, we want to find *any* solution.
            # A pure DFS might be slow. Let's use a randomized DFS.
            
            # Candidate generation:
            # We know the diffs so far.
            # New diffs will be x - a for a in current_row.
            
            # Let's try a window of candidates around the minimum possible.
            # This is a heuristic to keep the row "tight" (small scope contribution).
            
            # For now, let's just try a range. If max_val is small, it's fast.
            # If max_val is large, we should be careful.
            
            # Let's try up to 20 candidates per step to keep it fast, 
            # preferring smaller values.
            
            candidates = list(range(start_val, min(start_val + 20, max_val + 1)))
            rng.shuffle(candidates) # Randomize to escape local traps
            
            for v in candidates:
                new_diffs = {v - a for a in current_row}
                # Check against forbidden_diffs (from other rows)
                # And check against current_diffs (from this row so far)
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                # Recurse
                res = backtrack(pos + 1, current_row + [v], current_diffs | new_diffs)
                if res is not None:
                    return res
            return None
        
        # Start backtracking
        # current_row starts with [0]
        # current_diffs starts with empty set
        return backtrack(1, [0], set())

    # Main Loop: Local Search + Row Reconstruction
    max_iters = 100000
    iter_count = 0
    
    # We will try to reduce the scope.
    # Current best scope is best_scope.
    # We will try to find a solution with scope < best_scope.
    # We can do this by fixing the scope M and checking if a solution exists.
    # Or we can just run the search and see if we find a better one.
    
    # Let's use a simulated annealing style approach but with row reconstruction moves.
    
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        if iter_count % 1000 == 0:
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        # Choose a move type
        # 1. Local move (move one element)
        # 2. Row reconstruction
        
        # With high probability, do local move. With low prob, do reconstruction.
        # As we get stuck, increase reconstruction probability.
        
        # Let's alternate or use a counter.
        # Every 100 local moves, try one reconstruction.
        
        if iter_count % 100 == 0:
            # Try to reconstruct a random row
            r_idx = rng.randrange(n_rows)
            
            # What max_val to use for reconstruction?
            # We want to potentially reduce the scope.
            # Current scope of this row is rows[r_idx][-1].
            # Global scope is best_scope.
            # Let's try to reconstruct with max_val = best_scope - 1 (to improve global)
            # or max_val = rows[r_idx][-1] - 1 (to improve local)
            # Let's try to improve the global scope.
            
            # First, compute forbidden diffs from OTHER rows.
            forbidden = set()
            for i in range(n_rows):
                if i == r_idx:
                    continue
                row = rows[i]
                for a in row:
                    for b in row:
                        if a < b:
                            forbidden.add(b - a)
            
            # Try to reconstruct row r_idx with max_val = best_scope - 1
            # If successful, we have a solution with scope <= best_scope - 1.
            new_row = try_reconstruct_row(r_idx, forbidden, best_scope - 1, rows)
            
            if new_row is not None:
                # Check if it's actually better (it should be, since max_val < best_scope)
                # But we need to verify full validity just in case (though forbidden set covers others)
                # The new row's diffs are disjoint from forbidden by construction.
                # So the set is valid.
                rows[r_idx] = new_row
                new_scope = max(max(r) for r in rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                # Reset temperature? Or just continue.
                iter_count += 1
                continue
        
        # Else: Local Move
        # Pick a random row and a random element (not the first one)
        r_idx = rng.randrange(n_rows)
        pos = rng.randint(1, k) # indices 1..k
        old_val = rows[r_idx][pos]
        
        # Determine range for new value
        lo = rows[r_idx][pos - 1] + 1
        hi = rows[r_idx][pos + 1] - 1 if pos < k else best_scope + 10
        if hi <= lo:
            iter_count += 1
            continue
        
        # Pick a new value
        # Bias towards smaller values to reduce scope
        new_val = rng.randint(lo, hi)
        
        if new_val == old_val:
            iter_count += 1
            continue
            
        # Calculate change in differences
        # Remove old diffs involving this element
        # Add new diffs
        
        # For simplicity and robustness, we can just validate the whole set.
        # With k=8, n=4, total diffs ~ 144. Validation is O(n * k^2) ~ 4*64 = 256 ops.
        # This is fast enough for 100k+ iterations in Python?
        # 100k * 256 = 25M ops. In Python, maybe 1-2 seconds.
        # So we can afford full validation.
        
        rows[r_idx][pos] = new_val
        new_scope, _ = get_scope_and_diffs(rows)
        
        if new_scope is not None:
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                # Accept improvement
            else:
                # Accept if it's a hill climb (scope same or slightly worse?)
                # Usually we want to minimize scope.
                # Let's use a simple accept/reject.
                # If new_scope <= old_scope, accept.
                # If new_scope > old_scope, accept with probability exp(-delta/T).
                old_scope = max(max(r) for r in rows) # This is inefficient, we should track it
                # Let's just accept if new_scope <= best_scope or random < T
                # Actually, let's just accept if it's valid and (new_scope < best_scope OR rng.random() < T)
                # This allows exploring slightly worse solutions to escape minima.
                if new_scope < best_scope or rng.random() < T:
                    pass # Keep the change
                else:
                    rows[r_idx][pos] = old_val # Revert
        else:
            # Invalid
            rows[r_idx][pos] = old_val
            
        iter_count += 1

    return best_rows
