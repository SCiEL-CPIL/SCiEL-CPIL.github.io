import random
import time
import sys
from itertools import combinations
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # --- Helpers ---
    def get_scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def is_valid_solution(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def get_used_diffs(rows):
        """Get the set of all differences used in rows."""
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # --- Phase 1: Greedy Initialization ---
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

    current_rows = greedy_init()
    best_rows = [r[:] for r in current_rows]
    best_scope = get_scope(best_rows)

    # --- Phase 2: Pre-compute valid row patterns for clique approach ---
    # We'll try to find better solutions using a maximum clique approach
    # where nodes are valid row patterns and edges connect compatible patterns
    
    def generate_valid_rows(max_scope, limit=5000):
        """Generate valid row patterns (Sidon sequences of length k+1 starting at 0)."""
        valid_rows = []
        used_global = set()
        
        def backtrack(current_row, current_max, used_diffs):
            if len(current_row) == k + 1:
                valid_rows.append(current_row[:])
                return True
            
            start = current_row[-1] + 1 if current_row else 1
            for x in range(start, min(max_scope, current_max) + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if len(valid_rows) >= limit:
                    current_row.pop()
                    return True
                    
                backtrack(current_row, max_scope, used_diffs)
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        backtrack([], max_scope, set())
        return valid_rows

    # --- Phase 3: Local Search with Simulated Annealing ---
    def local_search(rows, deadline):
        """Improve rows using simulated annealing."""
        def calc_diff_counts(rows):
            diff_count = {}
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        diff_count[d] = diff_count.get(d, 0) + 1
            return diff_count

        current_rows = [r[:] for r in rows]
        current_diff_count = calc_diff_counts(current_rows)
        
        def count_conflicts(diff_count):
            return sum(c - 1 for c in diff_count.values() if c > 1)
            
        current_conflicts = count_conflicts(current_diff_count)
        current_scope = get_scope(current_rows)
        penalty = 100
        current_cost = current_scope + penalty * current_conflicts
        
        best_local = [r[:] for r in current_rows]
        best_local_scope = current_scope
        
        T = 5.0
        cooling = 0.999
        
        while time.time() < deadline:
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            j_idx = random.randint(1, k)
            
            old_val = row[j_idx]
            min_val = row[j_idx - 1] + 1
            if j_idx < k:
                max_val = row[j_idx + 1] - 1
            else:
                max_val = best_local_scope + 20
            
            if min_val > max_val:
                continue
                
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            # Update diff_count incrementally
            for i in range(j_idx):
                d = old_val - row[i]
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
                
            for i in range(j_idx + 1, len(row)):
                d = row[i] - old_val
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
                
            row[j_idx] = new_val
            
            for i in range(j_idx):
                d = new_val - row[i]
                current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
            for i in range(j_idx + 1, len(row)):
                d = row[i] - new_val
                current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
            new_conflicts = count_conflicts(current_diff_count)
            new_scope = get_scope(current_rows)
            new_cost = new_scope + penalty * new_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < pow(2.71828, -delta_cost / T):
                current_cost = new_cost
                current_conflicts = new_conflicts
                current_scope = new_scope
                
                if new_conflicts == 0 and new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local = [r[:] for r in current_rows]
            else:
                row[j_idx] = old_val
                for i in range(j_idx):
                    d = new_val - row[i]
                    current_diff_count[d] -= 1
                    if current_diff_count[d] == 0:
                        del current_diff_count[d]
                    
                for i in range(j_idx + 1, len(row)):
                    d = row[i] - new_val
                    current_diff_count[d] -= 1
                    if current_diff_count[d] == 0:
                        del current_diff_count[d]
                    
                for i in range(j_idx):
                    d = old_val - row[i]
                    current_diff_count[d] = current_diff_count.get(d, 0) + 1
                    
                for i in range(j_idx + 1, len(row)):
                    d = row[i] - old_val
                    current_diff_count[d] = current_diff_count.get(d, 0) + 1

            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_local

    # --- Phase 4: Clique-based search for better solutions ---
    def clique_search(target_scope, deadline, max_patterns=2000):
        """
        Find n_rows compatible patterns from pre-computed valid rows
        using a greedy clique approach.
        """
        # Generate valid rows within target scope
        patterns = generate_valid_rows(target_scope, max_patterns)
        
        if len(patterns) < n_rows:
            return None
        
        # Compute difference sets for each pattern
        diff_sets = []
        for p in patterns:
            ds = set()
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    ds.add(p[j] - p[i])
            diff_sets.append(ds)
        
        # Greedy clique: pick patterns that don't share differences
        selected = []
        selected_diffs = set()
        
        # Sort patterns by their maximum element (prefer smaller)
        sorted_indices = sorted(range(len(patterns)), key=lambda i: max(patterns[i]))
        
        for idx in sorted_indices:
            if len(selected) == n_rows:
                break
            if not (diff_sets[idx] & selected_diffs):
                selected.append(patterns[idx])
                selected_diffs |= diff_sets[idx]
        
        if len(selected) == n_rows:
            return selected
        
        return None

    # --- Main optimization loop ---
    # First, try local search on the greedy solution
    if time.time() < deadline - 10.0:
        improved = local_search(current_rows, deadline - 10.0)
        improved_scope = get_scope(improved)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved
    
    # Try clique-based search with decreasing target scopes
    if time.time() < deadline - 5.0:
        # Start from current best and try to reduce
        for target in range(best_scope - 1, max(50, best_scope - 30), -1):
            if time.time() > deadline - 5.0:
                break
            
            result = clique_search(target, deadline - 5.0)
            if result:
                result_scope = get_scope(result)
                if result_scope < best_scope:
                    best_scope = result_scope
                    best_rows = result
                    # Try to improve further with local search
                    if time.time() < deadline - 10.0:
                        improved = local_search(best_rows, deadline - 10.0)
                        improved_scope = get_scope(improved)
                        if improved_scope < best_scope:
                            best_scope = improved_scope
                            best_rows = improved
                break  # Only try one target scope per iteration
    
    # Final local search if time remains
    if time.time() < deadline - 2.0:
        final_improved = local_search(best_rows, deadline - 2.0)
        final_scope = get_scope(final_improved)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = final_improved
    
    # Verify the solution is valid
    if not is_valid_solution(best_rows):
        # Fallback to greedy if something went wrong
        best_rows = greedy_init()
    
    return best_rows
