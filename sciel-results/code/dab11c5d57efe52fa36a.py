import random
import time
import math
import sys
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    # Each row has k+1 entries (including 0). 
    # Number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting bound: max entry must be at least total_diffs
    lower_bound = total_diffs

    def get_max_entry(rows):
        m = 0
        for row in rows:
            if row and row[-1] > m:
                m = row[-1]
        return m

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(seed=None, target_max=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                # Try smallest available
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
                    if x > 5000: # Safety
                        break
                if ok:
                    row.append(x)
                    for a in row:
                        used.add(x - a)
                    found = True
                else:
                    # Fallback: just add x
                    row.append(x)
            rows.append(row)
        return rows

    # Initialize with a good greedy solution
    best_rows = greedy_construct(seed=42)
    if not is_valid(best_rows):
        # Fallback: very safe greedy
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
        best_rows = rows

    best_max = get_max_entry(best_rows)

    # New Strategy: Iterated Local Search (ILS) with aggressive perturbation
    # The key insight from the prompt: "Reducing per-restart time to 0.2s allows more restarts... increasing diversity"
    # We will implement a fast local search that runs for a short time, then perturbs, then repeats.
    # This is different from Tabu Search (which uses memory) or SA (which uses temperature).
    # ILS uses a "kick" to escape local minima.

    def build_diff_counts(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def count_conflicts(counts):
        conf = 0
        for c in counts.values():
            if c > 1:
                conf += c - 1
        return conf

    def fast_local_search(initial_rows, deadline):
        current_rows = [row[:] for row in initial_rows]
        curr_max = get_max_entry(current_rows)
        
        diff_counts = build_diff_counts(current_rows)
        curr_conflicts = count_conflicts(diff_counts)
        
        best_local_max = curr_max
        best_local_rows = [row[:] for row in current_rows]
        
        # Parameters for fast search
        max_iter = 5000 # Limit iterations to keep it fast
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if no_improve > 500:
                break # Stuck, exit early to allow perturbation

            # Pick a random move
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            row = current_rows[r_idx]
            old_val = row[c_idx]
            
            min_val = row[c_idx - 1] + 1
            max_val = curr_max + 10
            if c_idx < k:
                max_val = min(max_val, row[c_idx + 1] - 1)
            
            if min_val > max_val:
                continue
                
            # Bias towards smaller values to tighten the solution
            if random.random() < 0.8:
                new_val = random.randint(min_val, min(old_val, max_val))
            else:
                new_val = random.randint(min_val, max_val)
                
            if new_val == old_val:
                continue
            
            # Compute delta in conflicts
            old_diffs = []
            for i in range(len(row)):
                if i == c_idx: continue
                if i < c_idx: d = row[c_idx] - row[i]
                else: d = row[i] - row[c_idx]
                old_diffs.append(d)
                
            new_diffs = []
            for i in range(len(row)):
                if i == c_idx: continue
                if i < c_idx: d = new_val - row[i]
                else: d = row[i] - new_val
                new_diffs.append(d)
                
            # Calculate change in conflict count
            delta_conf = 0
            for d in old_diffs:
                if diff_counts.get(d, 0) > 1:
                    delta_conf -= 1
            for d in new_diffs:
                if diff_counts.get(d, 0) >= 1:
                    delta_conf += 1
            
            new_conflicts = curr_conflicts + delta_conf
            new_max = max(curr_max, new_val)
            if new_max == curr_max and new_val < old_val:
                if old_val == curr_max:
                    # Check if any other element is the max
                    temp_max = 0
                    for r in current_rows:
                        if r[-1] > temp_max: temp_max = r[-1]
                    # This is expensive, so only do it if necessary
                    # For now, approximate: if we moved the max down, new_max might be lower
                    # Let's just recompute max if old_val was the max
                    new_max = get_max_entry(current_rows) # Expensive but rare
            
            # Accept if it reduces conflicts, or if conflicts are 0 and max reduces
            if new_conflicts < curr_conflicts or (new_conflicts == 0 and new_max < curr_max):
                # Apply move
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] <= 0: del diff_counts[d]
                row[c_idx] = new_val
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                
                curr_conflicts = new_conflicts
                curr_max = new_max
                no_improve = 0
                
                if new_conflicts == 0 and new_max < best_local_max:
                    best_local_max = new_max
                    best_local_rows = [row[:] for row in current_rows]
            else:
                no_improve += 1

        return best_local_rows, best_local_max

    def perturb(rows, strength=0.1):
        """
        Perturb the rows by randomly changing some elements.
        This breaks the local structure to allow the next local search to explore a different basin.
        """
        perturbed = [row[:] for row in rows]
        n_changes = max(1, int(n_rows * k * strength))
        for _ in range(n_changes):
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            row = perturbed[r_idx]
            min_val = row[c_idx - 1] + 1
            max_val = row[c_idx] + 20 # Allow expansion
            if c_idx < k:
                max_val = min(max_val, row[c_idx + 1] - 1)
            if min_val <= max_val:
                row[c_idx] = random.randint(min_val, max_val)
        return perturbed

    # Main ILS Loop
    current_best_rows = best_rows
    current_best_max = best_max
    current_rows = best_rows[:]
    
    restarts = 0
    while time.time() < deadline - 5:
        # Run local search for a short burst
        found_rows, found_max = fast_local_search(current_rows, time.time() + 2.0)
        
        if found_max < current_best_max and is_valid(found_rows):
            current_best_max = found_max
            current_best_rows = found_rows
            current_rows = found_rows[:]
            # Reset perturbation strength or keep it
        else:
            # Perturb the current state (or the best state if we just got stuck)
            # We perturb the state that was input to local search, or the best state?
            # Usually, we perturb the current state.
            # If we improved, we might perturb the new best to explore further.
            # If we didn't improve, we perturb the current state to escape.
            
            # Let's perturb the best known state to generate a new diverse starting point
            # This is "Iterated Local Search" on the best solution
            current_rows = perturb(current_best_rows, strength=0.05 + 0.05 * (restarts % 5))
        
        restarts += 1
        if restarts % 10 == 0 and time.time() > deadline - 10:
            break

    # Final check
    if is_valid(current_best_rows):
        return current_best_rows
    else:
        # Fallback to a valid solution if somehow we lost validity
        return greedy_construct(seed=123)

