# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing on the full assignment with conflict-based moves.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 140.0  # safety margin before 150s kill
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    def validate(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    def compute_conflicts(rows):
        """Return dict mapping each difference to list of (row_idx, i, j) that produce it."""
        from collections import defaultdict
        diff_map = defaultdict(list)
        for ri, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_map[d].append((ri, i, j))
        conflicts = {}
        for d, locs in diff_map.items():
            if len(locs) > 1:
                conflicts[d] = locs
        return conflicts
    
    def compute_cost(rows):
        """Number of conflicting differences (each conflict pair counts once)."""
        cm = compute_conflicts(rows)
        cost = 0
        for d, locs in cm.items():
            # Each pair of entries in locs is a conflict
            cost += len(locs) * (len(locs) - 1) // 2
        return cost
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    # Simulated annealing
    def simulated_annealing(n_rows, k, max_bound, time_budget, seed=42):
        rng = random.Random(seed)
        deadline = time.time() + time_budget
        
        # Initialize with greedy, then try to compress
        rows = greedy_construct(n_rows, k)
        
        # Try to find a valid solution within max_bound
        # First, try direct greedy with bound
        def greedy_with_bound(n_rows, k, max_bound):
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    if x > max_bound:
                        return None
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
        
        greedy_b = greedy_with_bound(n_rows, k, max_bound)
        if greedy_b is not None and validate(greedy_b):
            return greedy_b
        
        # Start from greedy and try to reduce max entry
        current_rows = greedy_rows[:]
        # Scale down: try to fit within max_bound
        current_scale = max_entry(current_rows)
        if current_scale > max_bound:
            # Random perturbation to break symmetry
            for ri in range(n_rows):
                for j in range(1, len(current_rows[ri])):
                    if current_rows[ri][j] > max_bound:
                        current_rows[ri][j] = rng.randint(current_rows[ri][j-1]+1, max_bound)
            # Re-sort each row
            for ri in range(n_rows):
                current_rows[ri] = sorted(set(current_rows[ri]))
                # Ensure starts at 0 and has k+1 elements
                if current_rows[ri][0] != 0:
                    current_rows[ri] = [0] + current_rows[ri][1:k+1]
                if len(current_rows[ri]) != k+1:
                    # Pad
                    while len(current_rows[ri]) < k+1:
                        current_rows[ri].append(current_rows[ri][-1] + 1)
        
        current_cost = compute_cost(current_rows)
        current_max = max_entry(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        best_max = current_max
        
        # If already valid and within bound, return
        if current_cost == 0 and current_max <= max_bound:
            return current_rows
        
        T = 100.0  # initial temperature
        cooling = 0.995
        min_T = 0.01
        
        step_count = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and T > min_T:
            if step_count % 1000 == 0 and no_improve > max_no_improve:
                # Restart with new randomization
                current_rows = greedy_rows[:]
                for ri in range(n_rows):
                    for j in range(1, len(current_rows[ri])):
                        if current_rows[ri][j] > max_bound:
                            current_rows[ri][j] = rng.randint(current_rows[ri][j-1]+1, max_bound)
                for ri in range(n_rows):
                    current_rows[ri] = sorted(set(current_rows[ri]))
                    if current_rows[ri][0] != 0:
                        current_rows[ri] = [0] + current_rows[ri][1:k+1]
                    if len(current_rows[ri]) != k+1:
                        while len(current_rows[ri]) < k+1:
                            current_rows[ri].append(current_rows[ri][-1] + 1)
                current_cost = compute_cost(current_rows)
                current_max = max_entry(current_rows)
                no_improve = 0
                T = max(T, 50.0)
            
            # Pick a random row and position to modify
            ri = rng.randint(0, n_rows - 1)
            # Pick a position (not 0)
            pos = rng.randint(1, k)
            
            # Try to move this element
            old_val = current_rows[ri][pos]
            
            # Determine range for new value
            lo = current_rows[ri][pos-1] + 1
            hi = min(max_bound, current_rows[ri][pos+1] - 1) if pos < k else max_bound
            
            if lo > hi:
                T *= cooling
                step_count += 1
                continue
            
            # Choose new value: biased toward reducing conflicts
            # Try a few random candidates
            candidates = []
            num_tries = min(20, hi - lo + 1)
            for _ in range(num_tries):
                nv = rng.randint(lo, hi)
                if nv != old_val:
                    candidates.append(nv)
            
            if not candidates:
                T *= cooling
                step_count += 1
                continue
            
            new_val = rng.choice(candidates)
            
            # Apply move
            current_rows[ri][pos] = new_val
            # Re-sort the row (shouldn't be needed if we maintain order, but safe)
            # Actually we must maintain strict increase, so let's just check
            if pos > 0 and current_rows[ri][pos] <= current_rows[ri][pos-1]:
                current_rows[ri][pos] = old_val
                T *= cooling
                step_count += 1
                continue
            if pos < k and current_rows[ri][pos] >= current_rows[ri][pos+1]:
                current_rows[ri][pos] = old_val
                T *= cooling
                step_count += 1
                continue
            
            new_cost = compute_cost(current_rows)
            new_max = max_entry(current_rows)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                current_cost = new_cost
                current_max = new_max
                no_improve = 0
                if current_cost < best_cost or (current_cost == best_cost and current_max < best_max):
                    best_cost = current_cost
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            else:
                current_rows[ri][pos] = old_val
                no_improve += 1
            
            T *= cooling
            step_count += 1
        
        # Return best found
        if best_cost == 0 and best_max <= max_bound:
            return best_rows
        return None
    
    # Try to find solution for various max_entry values
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try from lb upward
    for max_entry in range(lb, greedy_max + 1):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 3
        
        if remaining < 1.0:
            break
        
        # Budget allocation
        distance = max_entry - lb
        if distance == 0:
            budget = min(20.0, remaining)
        elif distance <= 3:
            budget = min(15.0, remaining)
        elif distance <= 10:
            budget = min(8.0, remaining)
        else:
            budget = min(3.0, remaining)
        
        if budget < 0.5:
            continue
        
        # Multiple restarts
        num_restarts = max(1, int(budget / 4.0))
        restart_budget = budget / num_restarts
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 3:
                break
            result = simulated_annealing(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            break
    
    # Final validation
    if not validate(best_rows):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
