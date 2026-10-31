"""Difference triangle set (4,7) solver using Iterated Local Search with kick perturbations."""
import time
import random
import sys

# We must keep the entry point name and signature.
def construct(n_rows, k):
    # The task specifies (4,7) but the interface takes n_rows and k.
    # We assume the evaluator passes n_rows=4 and k=7.
    # If other values are passed, we try to solve them too, but optimize for 4x8.
    
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Pre-compute all pairs (i, j) for a row of size k+1
    # There are C(k+1, 2) differences per row.
    # For k=7, size=8, diffs=28.
    # Total diffs = n_rows * 28. For n_rows=4, total=112.
    
    size = k + 1
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    n_pairs = len(pairs)
    total_diffs = n_rows * n_pairs
    
    # Helper to extract differences from a row
    def get_diffs(row):
        return [row[j] - row[i] for i, j in pairs]
    
    # Helper to check validity
    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))
    
    # Helper to get scope
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Initial construction: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < size:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # If greedy is already very good, we might not need much search,
    # but we should still try to improve.
    
    # Iterated Local Search (ILS)
    # State: list of rows.
    # Move: pick a random row, pick a random position in that row, change the value.
    # Constraint: must remain strictly increasing and start at 0.
    # Objective: minimize scope (max element).
    # Feasibility: all differences distinct.
    
    # We will use a two-phase approach within ILS:
    # 1. Feasibility: If invalid, try to fix conflicts.
    # 2. Optimization: If valid, try to reduce scope.
    
    # Actually, a simpler and often effective method for this specific problem is
    # to use a local search that optimizes a penalty function:
    # Penalty = Scope + Lambda * (Number of Conflicting Differences)
    # We want to minimize this.
    
    # Let's implement a Simulated Annealing style local search on the penalty.
    
    def calc_penalty(rows):
        scope = get_scope(rows)
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        
        # Count conflicts
        # We want to know how many differences are duplicated.
        # A set of N items has C(N,2) pairs. If all unique, 0 conflicts.
        # If there are duplicates, we can count the number of "extra" items.
        # Or simply: total_items - unique_items.
        n_items = len(all_diffs)
        n_unique = len(set(all_diffs))
        conflicts = n_items - n_unique
        
        # We prioritize feasibility.
        # If conflicts > 0, the penalty should be dominated by conflicts.
        # Let's use: penalty = scope + 1000 * conflicts
        # This ensures we first find a valid solution, then minimize scope.
        
        return scope + 1000 * conflicts
    
    current_rows = [r[:] for r in best_rows]
    current_penalty = calc_penalty(current_rows)
    
    best_rows_il = [r[:] for r in current_rows]
    best_penalty_il = current_penalty
    best_scope_il = get_scope(best_rows_il)
    
    # If greedy is valid, we can start optimization immediately.
    # If not, we need to find a valid one.
    
    T = 100.0  # Initial temperature
    cooling = 0.999
    min_T = 0.01
    
    # To speed up, we can pre-calculate which differences are involved in conflicts?
    # That's complex. Let's just re-evaluate.
    # For 4x8, re-evaluation is fast (112 diffs).
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        # Pick a random position (not 0, as it must be 0)
        pos = random.randint(1, size - 1)
        
        old_val = current_rows[r_idx][pos]
        
        # Determine range for new value
        # Must be > current_rows[r_idx][pos-1]
        # Must be < current_rows[r_idx][pos+1] if pos < size-1
        lo = current_rows[r_idx][pos - 1] + 1
        hi = current_rows[r_idx][pos + 1] - 1 if pos < size - 1 else old_val + 10
        
        if lo > hi:
            # Try to expand? Or just skip.
            # If we can't move, try a different move.
            # For now, skip.
            continue
        
        # Random step size
        # Bias towards smaller steps?
        # Let's pick a new value uniformly in [lo, hi]
        # Or do a random walk step.
        # Uniform in [lo, hi] is a "kick".
        # Let's do a small random walk first, and occasionally a kick.
        
        if random.random() < 0.1:
            # Kick: random value in range
            new_val = random.randint(lo, hi)
        else:
            # Small step
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            if new_val < lo: new_val = lo
            if new_val > hi: new_val = hi
        
        if new_val == old_val:
            continue
        
        # Apply move
        current_rows[r_idx][pos] = new_val
        
        # Check strict increasing (already enforced by lo/hi logic mostly, but double check neighbors)
        if pos > 0 and current_rows[r_idx][pos] <= current_rows[r_idx][pos - 1]:
            current_rows[r_idx][pos] = old_val
            continue
        if pos < size - 1 and current_rows[r_idx][pos] >= current_rows[r_idx][pos + 1]:
            current_rows[r_idx][pos] = old_val
            continue
        
        new_penalty = calc_penalty(current_rows)
        delta_penalty = new_penalty - current_penalty
        
        if delta_penalty < 0 or random.random() < pow(2.718281828, -delta_penalty / T):
            current_penalty = new_penalty
            if new_penalty < best_penalty_il:
                best_penalty_il = new_penalty
                best_rows_il = [r[:] for r in current_rows]
                best_scope_il = get_scope(best_rows_il)
        else:
            # Revert
            current_rows[r_idx][pos] = old_val
        
        T *= cooling
        if T < min_T:
            # Restart with higher T (Iterated part)
            T = 10.0  # Lower than initial, to refine
            # Randomize current state slightly?
            # Or just reset to best?
            # Let's reset to best to avoid getting stuck in a bad local minimum.
            current_rows = [r[:] for r in best_rows_il]
            current_penalty = best_penalty_il
            
            # Add some noise to escape local minima (Kick)
            for _ in range(5):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, size - 1)
                lo = current_rows[r_idx][pos - 1] + 1
                hi = current_rows[r_idx][pos + 1] - 1 if pos < size - 1 else current_rows[r_idx][pos] + 10
                if lo <= hi:
                    current_rows[r_idx][pos] = random.randint(lo, hi)
            
            # Recalculate penalty after kick
            current_penalty = calc_penalty(current_rows)
            # It's possible the kick made it worse, that's fine, SA will try to fix it.
            # But if it's much worse, we might want to revert to best?
            # No, ILS usually keeps the kicked state and continues.
            
        # If we found a valid solution with scope <= 134, we can be more aggressive?
        # The known best is 134.
        # If we find < 134, that's a record.
        
        # Early exit if we hit the theoretical lower bound?
        # Lower bound is 112 (counting bound).
        # If best_scope_il == 112, we can stop.
        if best_scope_il == 112:
            break
    
    # Return the best valid solution found.
    # If best_rows_il is invalid (penalty > scope due to conflicts), we should return a valid one.
    # Our penalty function ensures valid solutions have lower penalty than invalid ones with same scope?
    # penalty = scope + 1000 * conflicts.
    # If conflicts > 0, penalty >= 1000.
    # Max scope is likely < 1000.
    # So any valid solution (conflicts=0) will have penalty < 1000.
    # Any invalid solution will have penalty >= 1000.
    # So best_penalty_il will correspond to a valid solution if one was found.
    
    # Check if best_rows_il is valid
    if is_valid(best_rows_il):
        return best_rows_il
    else:
        # Fallback to greedy if no valid solution was found during search (unlikely)
        # Or return the least invalid one? No, must be valid.
        # Greedy is always valid.
        return best_rows

# For testing purposes, we can run it directly if called as main
if __name__ == "__main__":
    # The evaluator calls construct(4, 7)
    # We can test here
    rows = construct(4, 7)
    for r in rows:
        print(r)
    diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                diffs.append(r[j]-r[i])
    print("Valid:", len(diffs) == len(set(diffs)))
    print("Scope:", max(max(r) for r in rows))
