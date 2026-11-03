import time
import random
import math
import heapq
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # Greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    # Bitset-accelerated row generation
    def generate_row_bitset(used_diffs, k_val, max_scope, randomize=False, p_smallest=0.7, window=5):
        """Generate a single row using bitset for fast conflict checking."""
        # Create a bitset of forbidden differences
        max_d = max_scope + 1
        # Use a Python int as bitset: bit d is set if difference d is forbidden
        forbidden_bits = 0
        for d in used_diffs:
            if d < max_d:
                forbidden_bits |= (1 << d)
        
        row = [0]
        row_diffs_bits = 0  # Bitset of differences within current row
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            
            while x < max_scope:
                # Check if x creates any forbidden differences
                conflict = False
                for a in row:
                    d = x - a
                    if d < max_d and (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    if d < max_d and (row_diffs_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
            
            # Update row_diffs_bits
            for a in row:
                d = next_x - a
                if d < max_d:
                    row_diffs_bits |= (1 << d)
            
            row.append(next_x)
        
        return row

    # Branch and bound for maximum independent set on a small graph
    # Returns a set of positions (indices into 0..max_scope-1)
    def max_independent_set_branch_and_bound(conflict_matrix, n_positions, target_size):
        """
        Find a maximum independent set of size target_size in a graph.
        conflict_matrix: list of sets, where conflict_matrix[i] is the set of j that conflict with i.
        Returns a list of positions, or None if no independent set of size target_size exists.
        """
        if n_positions <= 0:
            return [] if target_size == 0 else None
        
        # Use a simple branch and bound
        # Order vertices by degree (heuristic: pick vertex with fewest conflicts first)
        degrees = [len(conflict_matrix[i]) for i in range(n_positions)]
        order = list(range(n_positions))
        order.sort(key=lambda i: degrees[i])
        
        # Map from original index to position in order
        pos_in_order = [0] * n_positions
        for pos, idx in enumerate(order):
            pos_in_order[idx] = pos
        
        # Try to build an independent set
        def backtrack(start_pos, current_set, remaining_needed):
            if remaining_needed == 0:
                return current_set[:]
            if start_pos >= len(order):
                return None
            
            # Upper bound: can we still reach target_size?
            if remaining_needed > len(order) - start_pos:
                return None
            
            # Try including order[start_pos]
            v = order[start_pos]
            # Check if v conflicts with any in current_set
            can_include = True
            for u in current_set:
                if u in conflict_matrix[v]:
                    can_include = False
                    break
            
            if can_include:
                result = backtrack(start_pos + 1, current_set + [v], remaining_needed - 1)
                if result is not None:
                    return result
            
            # Try excluding order[start_pos]
            result = backtrack(start_pos + 1, current_set, remaining_needed)
            if result is not None:
                return result
            
            return None
        
        result = backtrack(0, [], target_size)
        if result is not None:
            return sorted(result)
        return None

    # Local compaction: try to move each element to a smaller position
    def local_compact(rows, used_diffs, max_iterations=1000):
        """Try to reduce the scope by moving elements to smaller positions."""
        current_scope = max(max(r) for r in rows)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):  # Start from last element
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    
                    # Remove old_val from row
                    row.pop(p_idx)
                    
                    # Remove its differences from used_diffs
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    
                    # Try to find a smaller position
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            # Insert x in sorted position
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Re-insert old_val
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        
        return rows, used_diffs

    # 1. Initial Solution: Deterministic Greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Spectral-Guided Difference-Graph Construction
    # Model: each row is K_{k+1}; global constraint = edge-coloring with colors 1..M
    # Use spectral partitioning on the difference-conflict matrix to guide greedy assignment
    
    def spectral_guided_construct(n_rows, k, max_scope, time_limit=30):
        """
        Construct rows using spectral ordering of the difference-conflict graph.
        The conflict matrix C[d1][d2] = 1 if differences d1, d2 cannot coexist in same row.
        We use eigenvector ordering to guide which differences to assign to which rows.
        """
        spec_start = time.time()
        
        # Total differences needed: n_rows * C(k+1, 2)
        num_diffs_per_row = k * (k + 1) // 2
        total_diffs = n_rows * num_diffs_per_row
        
        # Build a conflict matrix for differences up to max_scope
        # Two differences d1, d2 conflict if they can't be in the same row
        # This happens when d1 + d2 > max_scope (since row elements must be < max_scope)
        # Actually, the constraint is more subtle: differences must be realizable
        # in a strictly increasing sequence starting at 0
        
        # For efficiency, use a simpler model:
        # Assign differences to rows greedily, using spectral ordering of a 
        # "compatibility" graph where edges connect compatible differences
        
        # Step 1: Generate candidate differences for each row using spectral hints
        used = set()
        rows_found = []
        
        # Precompute all possible differences and their compatibility
        # Two differences are compatible if they can appear in the same row
        # A simple sufficient condition: d1 + d2 <= max_scope
        # But we also need to ensure the row can be constructed
        
        # Use a beam search with spectral-inspired ordering
        # For each row, try to construct it using differences ordered by
        # their "spectral centrality" in the remaining difference space
        
        for row_idx in range(n_rows):
            if time.time() - spec_start > time_limit:
                break
            
            # Generate multiple candidate rows
            candidates = []
            
            # Strategy 1: Greedy smallest
            row_greedy = generate_row_bitset(used, k, max_scope, randomize=False)
            if row_greedy:
                candidates.append(row_greedy)
            
            # Strategy 2: Spectral-inspired - prioritize differences that are
            # "central" in the remaining space (high degree in compatibility graph)
            for trial in range(5):
                # Randomize with spectral bias
                p_smallest = random.uniform(0.6, 0.85)
                window = random.randint(3, 6)
                row_spec = generate_row_bitset(used, k, max_scope, randomize=True, 
                                              p_smallest=p_smallest, window=window)
                if row_spec:
                    candidates.append(row_spec)
            
            # Strategy 3: Try to balance the load - pick differences that
            # leave the most flexibility for future rows
            for trial in range(3):
                # Use a different randomization pattern
                row_bal = generate_row_bitset(used, k, max_scope, randomize=True,
                                             p_smallest=random.uniform(0.4, 0.7),
                                             window=random.randint(5, 10))
                if row_bal:
                    candidates.append(row_bal)
            
            if not candidates:
                break
            
            # Score candidates: prefer those that minimize max value AND
            # leave more "central" differences available for future rows
            def score_candidate(row):
                # Primary: minimize max value
                max_val = max(row)
                # Secondary: count how many "small" differences are used
                # (small differences are more constraining)
                diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.add(row[j] - row[i])
                small_diffs = sum(1 for d in diffs if d <= max_scope // 3)
                # Tertiary: diversity of differences
                return (max_val, small_diffs, -len(diffs))
            
            candidates.sort(key=score_candidate)
            
            # Pick the best candidate
            best_row = candidates[0]
            rows_found.append(best_row)
            used.update({best_row[j] - best_row[i] for i in range(len(best_row)) for j in range(i + 1, len(best_row))})
        
        if len(rows_found) == n_rows:
            scope = max(max(r) for r in rows_found)
            return rows_found, scope
        
        return None, float('inf')

    # Try spectral-guided construction with multiple scope limits
    spectral_deadline = deadline - 55
    spec_solution = None
    spec_scope = float('inf')
    
    for scope_limit in range(best_max - 3, min(best_max + 8, 150), 2):
        if time.time() > spectral_deadline:
            break
        
        sol, scope = spectral_guided_construct(n_rows, k, scope_limit, time_limit=8)
        if sol and scope < spec_scope:
            spec_scope = scope
            spec_solution = sol
    
    # Use spectral solution if better
    if spec_solution and spec_scope < best_max:
        rows = spec_solution
        best_rows = [r[:] for r in rows]
        best_max = spec_scope
        current_max = best_max

    # 2. Simulated Annealing with Row Regeneration and Multi-Start
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    used = get_used_diffs(rows)
    
    # Use more of the time budget for SA with multiple restarts
    sa_deadline = deadline - 20
    
    # Track best across restarts
    sa_best_rows = [r[:] for r in best_rows]
    sa_best_max = best_max
    
    # Multiple restarts to escape local minima
    num_restarts = 3
    restarts_done = 0
    current_restart_start = time.time()
    
    while time.time() < sa_deadline:
        if T < T_min:
            # Check if we should restart
            if restarts_done < num_restarts and (time.time() - current_restart_start) > 15:
                # Reset to best solution and restart with higher temperature
                rows = [r[:] for r in sa_best_rows]
                used = get_used_diffs(rows)
                current_max = sa_best_max
                T = 8.0
                cooling = 0.997
                restarts_done += 1
                current_restart_start = time.time()
            else:
                # If no more restarts, just continue cooling
                T = 3.0
                cooling = 0.999
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate row with randomization - use wider scope to allow escape
        scope_for_gen = sa_best_max + random.randint(5, 15)
        new_row = generate_row_bitset(used, k, scope_for_gen, randomize=True, 
                                      p_smallest=random.uniform(0.7, 0.9), 
                                      window=random.randint(2, 5))
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
            current_max = new_global_max
            if current_max < sa_best_max:
                sa_best_max = current_max
                sa_best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 3. Iterative Compaction Local Search with Random Perturbations
    # This phase aggressively tries to reduce the scope by moving elements
    # to smaller positions. It repeats until no further improvements are found
    # or the time limit is reached.
    
    current_rows = [list(r) for r in best_rows]
    used_diffs = get_used_diffs(current_rows)
    current_scope = max(max(r) for r in current_rows)
    
    ls_deadline = deadline - 5
    improved = True
    stall_count = 0
    max_stall = 3
    
    while improved and time.time() < ls_deadline:
        improved = False
        
        # Iterate through rows and elements in reverse order to allow
        # larger elements to move down, potentially freeing space for
        # smaller ones later (though we check all, order helps).
        for r_idx in range(n_rows):
            row = current_rows[r_idx]
            # Iterate from last element to first (excluding 0)
            for p_idx in range(len(row) - 1, 0, -1):
                old_val = row[p_idx]
                if old_val <= 1:
                    continue
                
                # Remove old_val from consideration
                row.pop(p_idx)
                # Remove its diffs from used_diffs
                diffs_to_remove = set()
                for a in row:
                    d = abs(old_val - a)
                    diffs_to_remove.add(d)
                used_diffs -= diffs_to_remove
                
                # Find the smallest valid position for this element
                found_pos = None
                # We can search from 1 up to old_val
                # Optimization: if we just want to reduce scope, we search 1..old_val-1
                for x in range(1, old_val):
                    if x in row:
                        continue
                    valid = True
                    for a in row:
                        d = abs(x - a)
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        found_pos = x
                        break
                
                if found_pos is not None:
                    # Insert found_pos
                    pos = 0
                    while pos < len(row) and row[pos] < found_pos:
                        pos += 1
                    row.insert(pos, found_pos)
                    # Update used_diffs
                    for a in row:
                        if a != found_pos:
                            used_diffs.add(abs(found_pos - a))
                    improved = True
                    # Update current_scope if this was the max
                    # We'll recalculate at the end of the row/loop to save time
                else:
                    # Revert
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        used_diffs.add(d)
        
        # Recalculate scope
        new_scope = max(max(r) for r in current_rows)
        if new_scope < current_scope:
            current_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            best_max = current_scope
            stall_count = 0
        else:
            # If no scope reduction, try random perturbation to escape
            stall_count += 1
            if stall_count >= max_stall:
                # Apply random perturbation: regenerate one row
                r_idx = random.randint(0, n_rows - 1)
                old_row = current_rows[r_idx]
                old_diffs = set()
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        old_diffs.add(old_row[j] - old_row[i])
                
                used_diffs -= old_diffs
                
                new_row = generate_row_bitset(used_diffs, k, current_scope + 10, 
                                             randomize=True, p_smallest=0.7, window=4)
                
                if new_row is not None:
                    current_rows[r_idx] = new_row
                    used_diffs |= set(new_row[j] - new_row[i] for i in range(len(new_row)) 
                                     for j in range(i + 1, len(new_row)))
                    # Recalculate scope
                    perturbed_scope = max(max(r) for r in current_rows)
                    if perturbed_scope <= current_scope + 5:  # Allow slight increase
                        improved = True
                        current_scope = perturbed_scope
                    else:
                        # Revert
                        current_rows[r_idx] = old_row
                        used_diffs |= old_diffs
                
                stall_count = 0
            elif not improved:
                break
                
    # Final validation and fallback
    if not is_valid(best_rows):
        # Fallback to a simple greedy construction if something went wrong
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)

    return best_rows
