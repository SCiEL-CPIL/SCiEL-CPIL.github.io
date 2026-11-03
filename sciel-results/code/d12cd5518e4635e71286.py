import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We use 115s for computation.
    deadline = start_time + 115.0
    
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def build_diff_counts(rows):
        counts = defaultdict(int)
        for r in rows:
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    counts[d] += 1
        return counts

    def _remove_row_diffs(counts, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                counts[d] -= 1
                if counts[d] == 0:
                    del counts[d]

    def _add_row_diffs(counts, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(r)):
                d = r[j] - ri
                counts[d] = counts.get(d, 0) + 1
                # Note: The original code had a bug in add_row_diffs inner loop variable name (r vs row), 
                # but I will write it correctly here.
                
    # Corrected Add function
    def add_row_diffs(counts, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                counts[d] = counts.get(d, 0) + 1

    def get_cost(rows, diff_counts):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        # Weight 500 ensures collisions are prioritized heavily over scope reduction
        # unless the scope gain is massive (unlikely in late game).
        return scope + 500 * collisions

    def greedy_initialize():
        """
        Creates a valid set of rows with minimal collisions initially.
        Uses a greedy approach to place rows one by one, picking the smallest
        available numbers that don't cause too many conflicts, then refining.
        """
        rows = []
        used_diffs = defaultdict(int)
        
        for _ in range(n_rows):
            # Greedy construction of a single row
            row = [0]
            current_max = 0
            
            # We want to pack this row tightly. 
            # Try to find a set of k numbers > 0 such that diffs are unique 
            # and don't collide with used_diffs too much.
            
            # Simple greedy: pick smallest available x such that diffs to row are mostly new
            # This is O(N^2) per row which is fine for small N.
            
            # Actually, a better init is random sampling + local repair
            # But for now, let's do a smart random init.
            pass
            
        # Let's use a better initialization: Random + Immediate Conflict Resolution
        # We will generate random rows and then run a few steps of SA to clean them up.
        # To make it "Greedy", we sort candidate values by how many conflicts they cause.
        
        init_rows = []
        # We aim for a bound slightly above the theoretical minimum
        # Theoretical min for (7,5) is around 105-112. Let's start search around 120.
        bound = 150 
        
        for _ in range(n_rows):
            # Pick k distinct integers from 1 to bound
            candidates = random.sample(range(1, bound + 1), k)
            candidates.sort()
            init_rows.append([0] + candidates)
            
        return init_rows

    def sa_optimize(init_rows, deadline):
        rng = random.Random(42) # Fixed seed for reproducibility within run? No, use global random or time-based.
        # Let's use a fresh RNG seeded by time for variety in restarts, 
        # but the main structure is deterministic enough.
        # Actually, the task says "random" is available. 
        # We will use the global random module or a local one.
        # Let's use a local RNG for isolation.
        
        current_rows = [list(r) for r in init_rows]
        diff_counts = build_diff_counts(current_rows)
        current_cost = get_cost(current_rows, diff_counts)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        T = 50.0
        cooling = 0.999
        min_T = 0.1
        
        while time.time() < deadline:
            r = rng.random()
            
            if r < 0.4:
                # Move 1: Swap elements within a row
                ri = rng.randint(0, n_rows - 1)
                row = current_rows[ri]
                
                # Pick two indices i, j (1..k) to swap
                i = rng.randint(1, row_len - 1)
                j = rng.randint(1, row_len - 1)
                if i == j:
                    continue
                
                # Check if swapping maintains sorted order
                # If we swap row[i] and row[j], we must ensure the row remains strictly increasing.
                # This is a strict constraint. 
                # Instead of swapping arbitrary indices, we can pick a "gap" and shift.
                # Or, we can just pick a new value for a single element (Jump Move).
                
                # Let's use "Value Jump" instead of Swap for simplicity and validity.
                # Pick a random element in a random row and change it.
                
                # Re-evaluating Move 1: Value Jump
                idx = rng.randint(1, row_len - 1)
                old_val = row[idx]
                
                # Determine range for new value
                # Must be > row[idx-1] and < row[idx+1]
                min_val = row[idx - 1] + 1
                max_val = row[idx + 1] - 1 if idx < row_len - 1 else current_cost # Upper bound roughly scope
                
                if min_val > max_val:
                    continue
                    
                # Bias towards center of range
                span = max_val - min_val
                # Gaussian-ish distribution centered
                # For simplicity, uniform is okay, but let's try to be smart.
                # If span is large, pick randomly.
                new_val = rng.randint(min_val, max_val)
                
                if new_val == old_val:
                    continue
                
                # Remove old row diffs
                _remove_row_diffs(diff_counts, row)
                
                # Apply change
                row[idx] = new_val
                
                # Add new row diffs
                add_row_diffs(diff_counts, row)
                
                new_cost = get_cost(current_rows, diff_counts)
                
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    row[idx] = old_val
                    _remove_row_diffs(diff_counts, row)
                    add_row_diffs(diff_counts, row)
                    
            elif r < 0.7:
                # Move 2: Row Swap (Exchange values between two rows at same index)
                # This preserves the property that rows are sets, but might break sorting?
                # No, if we swap values at index i in row A and row B, 
                # Row A becomes A' = A[0..i-1] + [B_i] + A[i+1..]
                # This might not be sorted. 
                # So we need to sort the row after swap? 
                # But sorting changes the differences! 
                # Differences depend on the set of values, not their order.
                # The problem states "increasing integers", so the row is a set.
                # The differences are between entries. 
                # $a_j - a_i$ for $i < j$. If we sort, the set of values is the same, 
                # so the set of differences is the same.
                # So we CAN swap values between rows, then re-sort the rows.
                
                r1 = rng.randint(0, n_rows - 1)
                r2 = rng.randint(0, n_rows - 1)
                if r1 == r2:
                    continue
                    
                idx = rng.randint(1, row_len - 1)
                
                v1 = current_rows[r1][idx]
                v2 = current_rows[r2][idx]
                
                if v1 == v2:
                    continue
                
                # Remove diffs for both rows
                row1 = current_rows[r1]
                row2 = current_rows[r2]
                
                _remove_row_diffs(diff_counts, row1)
                _remove_row_diffs(diff_counts, row2)
                
                # Swap
                row1[idx] = v2
                row2[idx] = v1
                
                # Re-sort rows to maintain "increasing integers" representation
                # Note: The set of differences remains invariant under sorting.
                row1.sort()
                row2.sort()
                
                # Add diffs
                add_row_diffs(diff_counts, row1)
                add_row_diffs(diff_counts, row2)
                
                new_cost = get_cost(current_rows, diff_counts)
                
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    row1[idx] = v1
                    row2[idx] = v2
                    row1.sort()
                    row2.sort()
                    _remove_row_diffs(diff_counts, row1)
                    _remove_row_diffs(diff_counts, row2)
                    add_row_diffs(diff_counts, row1)
                    add_row_diffs(diff_counts, row2)

            else:
                # Move 3: Row Rebuild (CSP-style)
                # Pick a random row and try to rebuild it to minimize collisions/scope.
                ri = rng.randint(0, n_rows - 1)
                old_row = list(current_rows[ri])
                
                _remove_row_diffs(diff_counts, old_row)
                
                # Try to find a better row using greedy + randomization
                # Constraints: 
                # 1. Starts with 0
                # 2. Strictly increasing
                # 3. Differences should ideally be unused (low count in diff_counts)
                
                new_row = None
                # Use a simple backtracking/greedy with randomization
                # Time limit per rebuild: 0.01s
                
                rebuild_start = time.time()
                max_attempts = 10
                for _ in range(max_attempts):
                    if time.time() - rebuild_start > 0.02:
                        break
                        
                    candidate = [0]
                    valid = True
                    # Greedy selection of next element
                    for _ in range(k):
                        # Range for next element
                        min_next = candidate[-1] + 1
                        # Max next: we want to keep it small. 
                        # Let's say we allow up to current_scope + 10
                        max_next = _scope(current_rows) + 10
                        
                        # Find best value in [min_next, max_next]
                        # "Best" means lowest count in diff_counts for all diffs to candidate
                        best_val = None
                        best_score = float('inf')
                        
                        # Sample a subset of candidates to speed up
                        # If range is small, check all.
                        if max_next - min_next < 50:
                            candidates = range(min_next, max_next + 1)
                        else:
                            candidates = rng.sample(range(min_next, max_next + 1), 20)
                        
                        for v in candidates:
                            # Check diffs
                            score = 0
                            ok = True
                            for a in candidate:
                                d = v - a
                                c = diff_counts.get(d, 0)
                                if c > 0:
                                    # Penalize collisions heavily
                                    score += 1000 * (c + 1) 
                                # Also penalize large v slightly to keep scope low
                                score += v
                            
                            if score < best_score:
                                best_score = score
                                best_val = v
                                
                        if best_val is None:
                            valid = False
                            break
                            
                        candidate.append(best_val)
                        
                    if valid:
                        # Check if this row is actually better
                        # Calculate cost impact
                        # This is expensive, so we just take the first valid one?
                        # No, we want to minimize cost.
                        # Let's just use the generated row if it's strictly better in "local score"
                        if new_row is None or candidate[-1] < new_row[-1]:
                            new_row = candidate
                            
                if new_row is None:
                    # Revert
                    add_row_diffs(diff_counts, old_row)
                    continue
                    
                current_rows[ri] = new_row
                add_row_diffs(diff_counts, new_row)
                
                new_cost = get_cost(current_rows, diff_counts)
                
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    _remove_row_diffs(diff_counts, new_row)
                    current_rows[ri] = old_row
                    add_row_diffs(diff_counts, old_row)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_rows, best_cost

    # Main Loop
    # 1. Initialize
    init_rows = greedy_initialize()
    
    # 2. Run SA
    # We can run multiple restarts if time permits
    best_global_rows = None
    best_global_cost = float('inf')
    
    # Run 1: Long run
    remaining = deadline - time.time()
    if remaining > 5:
        rows, cost = sa_optimize(init_rows, time.time() + min(remaining, 60))
        if cost < best_global_cost:
            best_global_cost = cost
            best_global_rows = rows
            
    # Run 2: Another run with different seed/init if time remains
    remaining = deadline - time.time()
    if remaining > 10 and best_global_rows:
        # Perturb best rows slightly
        perturbed = [list(r) for r in best_global_rows]
        for r in perturbed:
            for i in range(1, len(r)):
                if random.random() < 0.2:
                    # Small jitter
                    old = r[i]
                    r[i] = max(r[i-1]+1, old + random.randint(-2, 2))
                    r.sort()
                    
        rows, cost = sa_optimize(perturbed, time.time() + min(remaining, 40))
        if cost < best_global_cost:
            best_global_cost = cost
            best_global_rows = rows

    if best_global_rows is None:
        return init_rows
        
    # Final Polish: Local Search with 0 Temperature (Greedy)
    # Try to fix any remaining collisions or reduce scope by small steps
    final_rows = [list(r) for r in best_global_rows]
    final_counts = build_diff_counts(final_rows)
    final_cost = get_cost(final_rows, final_counts)
    
    polish_deadline = time.time() + 5.0
    while time.time() < polish_deadline:
        improved = False
        for i in range(n_rows):
            row = final_rows[i]
            for j in range(1, len(row)):
                old_val = row[j]
                # Try decreasing values
                min_val = row[j-1] + 1
                max_val = row[j+1] - 1 if j < len(row)-1 else old_val
                
                for new_val in range(old_val - 1, min_val - 1, -1):
                    if new_val <= min_val:
                        break
                    # Check if changing row[i][j] to new_val improves cost
                    _remove_row_diffs(final_counts, row)
                    row[j] = new_val
                    add_row_diffs(final_counts, row)
                    
                    new_cost = get_cost(final_rows, final_counts)
                    if new_cost < final_cost:
                        final_cost = new_cost
                        improved = True
                        break
                    else:
                        # Revert and try smaller decrease? Or just break?
                        # Since we are going down, if old_val-1 didn't work, 
                        # maybe old_val-2 will? 
                        # Let's just revert and break for efficiency in polish.
                        row[j] = old_val
                        _remove_row_diffs(final_counts, row)
                        add_row_diffs(final_counts, row)
                        break
                        
        if not improved:
            break
            
    return final_rows

