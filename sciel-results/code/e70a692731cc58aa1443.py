"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 145.0

    def max_entry(rows):
        return max(max(row) for row in rows)

    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 10000 + m

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # --- Phase 2: CSP with bitset-based backtracking ---
    # We use a bitset (Python int) to track used differences.
    # diff d is used iff (used_diffs >> d) & 1 == 1
    # Maximum difference is bounded by the current best_max.
    
    def try_backtracking(max_scope, timeout=5.0):
        """Try to find a valid configuration with max entry <= max_scope using backtracking."""
        start_time = time.time()
        used_diffs = 0  # bitset
        rows = [[0] for _ in range(n_rows)]
        # We'll fill entries one by one, row by row, position by position.
        # Order: row 0 pos 1, row 0 pos 2, ..., row 0 pos k, row 1 pos 1, ...
        
        # To make it faster, we can try to fill all rows' position 1 first, then position 2, etc.
        # This might help with symmetry breaking.
        
        # Let's use a recursive approach with memoization on (row_idx, pos_idx, used_diffs)
        # but used_diffs can be huge, so no memoization.
        
        # Instead, we'll use iterative deepening with a time limit.
        
        # Actually, let's just do a simple DFS with pruning.
        
        def dfs(row_idx, pos_idx):
            """Try to fill row row_idx at position pos_idx (1-indexed)."""
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                # All rows filled
                return True
            
            if pos_idx == k + 1:
                # This row is complete, move to next row
                return dfs(row_idx + 1, 1)
            
            row = rows[row_idx]
            prev_val = row[pos_idx - 1]
            
            # Try values from prev_val+1 to max_scope
            # But we need to ensure the row can still be completed.
            # Lower bound: we need k - pos_idx more entries, each at least +1 apart,
            # so the last entry must be at least prev_val + (k - pos_idx + 1)
            # Wait, pos_idx is 1-indexed, so we've placed pos_idx entries (including 0 at pos 0).
            # We need to place positions pos_idx, pos_idx+1, ..., k.
            # That's k - pos_idx + 1 more entries.
            # The minimum possible value for position pos_idx is prev_val + 1.
            # The minimum possible value for position k is prev_val + (k - pos_idx + 1).
            
            # For each candidate value x for position pos_idx:
            # - x > prev_val
            # - x <= max_scope
            # - The new differences {x - row[i] for i in 0..pos_idx-1} must not be in used_diffs
            # - We must be able to complete the row with remaining positions
            
            # To speed up, try smaller values first (to keep max small)
            # But also try to be smart about which values to try.
            
            # Heuristic: try values that don't conflict, starting from smallest.
            candidates = []
            for x in range(prev_val + 1, max_scope + 1):
                # Check if the new differences conflict
                new_diffs = {x - row[i] for i in range(pos_idx)}
                conflict = False
                for d in new_diffs:
                    if d > max_scope:  # Shouldn't happen since x <= max_scope and row[i] >= 0
                        conflict = True
                        break
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
            
            # If no candidates, backtrack
            if not candidates:
                return False
            
            # Try candidates in random order to break symmetry
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() - start_time > timeout:
                    return False
                
                # Place x
                row.append(x)
                
                # Update used_diffs
                new_diffs = {x - row[i] for i in range(pos_idx)}
                new_used = used_diffs
                for d in new_diffs:
                    new_used |= (1 << d)
                
                # Recurse
                if dfs(row_idx, pos_idx + 1):
                    return True
                
                # Backtrack
                row.pop()
                # used_diffs is restored by the caller's frame (it's a local variable in the closure)
                # Actually, we need to be careful. Let's pass used_diffs as a parameter.
            
            return False
        
        # The above has a bug: used_diffs is not properly backtracked because it's an int (immutable).
        # We need to pass it as a parameter. Let me rewrite.
        
        def dfs2(row_idx, pos_idx, used_diffs):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == k + 1:
                return dfs2(row_idx + 1, 1, used_diffs)
            
            row = rows[row_idx]
            prev_val = row[pos_idx - 1]
            
            for x in range(prev_val + 1, max_scope + 1):
                if time.time() - start_time > timeout:
                    return False
                
                new_diffs = {x - row[i] for i in range(pos_idx)}
                conflict = False
                new_used = used_diffs
                for d in new_diffs:
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                    new_used |= (1 << d)
                
                if conflict:
                    continue
                
                # Place x
                row.append(x)
                
                if dfs2(row_idx, pos_idx + 1, new_used):
                    return True
                
                # Backtrack
                row.pop()
            
            return False
        
        # Start DFS
        if dfs2(0, 1, 0):
            return [r[:] for r in rows]
        
        return None

    # --- Phase 3: Simulated Annealing ---
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    current_cost = compute_cost(current_rows)

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    # Try to improve via backtracking with decreasing scope
    # Start from best_max - 1 and go down
    for scope in range(best_max - 1, max(110, best_max - 20), -1):
        if time.time() > deadline:
            break
        
        result = try_backtracking(scope, timeout=3.0)
        if result is not None:
            # Found a valid solution with max entry = scope
            best_rows = result
            best_max = scope
            current_rows = [r[:] for r in best_rows]
            current_cost = compute_cost(current_rows)
            # Continue trying to go lower
            continue
        else:
            # Couldn't find a solution with this scope, try SA to improve
            break

    # Final SA polish
    T = 20.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 30
        
        if random.random() < 0.4:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.4)
        else:
            new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            continue
        
        current_rows[r][i] = new_val
        
        row = current_rows[r]
        if any(row[j] <= row[j-1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue
        
        new_cost = compute_cost(current_rows)
        
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            current_rows[r][i] = old_val
        
        T *= cooling

    return best_rows
# EVOLVE-BLOCK-END
