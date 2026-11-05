import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task says 120s. Use 105s to be safe.
    deadline = start_time + 105
    
    # 1. Greedy Initialization
    def greedy_init(seed=None):
        rng = random.Random(seed) if seed is not None else random
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    # 2. Compute Initial State
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    def compute_energy(M, penalty, T):
        if penalty == 0:
            return float(M)
        lambda_pen = 100.0 * (1.0 + T / 50.0)
        return M + lambda_pen * penalty

    def get_conflict_elements(counts, rows):
        """Return list of (row_idx, elem_idx) for elements involved in conflicts."""
        conflicts = {}
        for r_idx, r in enumerate(rows):
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if counts.get(d, 0) > 1:
                        conflicts.setdefault(r_idx, set()).add(i)
                        conflicts.setdefault(r_idx, set()).add(j)
        result = []
        for r_idx, elems in conflicts.items():
            for e in elems:
                if e > 0:  # index 0 is fixed
                    result.append((r_idx, e))
        return result

    # Multi-start SA
    best_rows = None
    best_M = float('inf')
    
    # Number of restarts - use more shorter restarts
    max_restarts = 8
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        # Check if we have enough time for this restart
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            curr_rows = greedy_init()
        else:
            # Random perturbation of the best solution found so far, or fresh greedy
            if best_rows is not None and random.random() < 0.4:
                curr_rows = copy.deepcopy(best_rows)
                # Perturb some elements more aggressively
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    j_idx = random.randint(1, k)
                    row = curr_rows[r_idx]
                    low = row[j_idx - 1] + 1
                    high = row[j_idx + 1] - 1 if j_idx < k else get_M(curr_rows) + 15
                    if high >= low:
                        row[j_idx] = random.randint(low, high)
                        # Fix ordering if violated
                        for i in range(1, k + 1):
                            if row[i] <= row[i-1]:
                                row[i] = row[i-1] + 1
            else:
                curr_rows = greedy_init()
        
        curr_counts = get_diff_counts(curr_rows)
        curr_penalty = get_penalty(curr_counts)
        curr_M = get_M(curr_rows)
        
        # Track M incrementally
        row_maxes = [max(r) for r in curr_rows]
        
        # SA Parameters - faster cooling for more exploration
        T = 80.0
        cooling = 0.9995
        curr_energy = compute_energy(curr_M, curr_penalty, T)
        
        # Local restart deadline
        local_deadline = start_time + (restart + 1) * time_per_restart
        
        # SA Loop
        while time.time() < local_deadline:
            # Generate Move - prioritize conflict resolution when conflicts exist
            if curr_penalty > 0 and random.random() < 0.6:
                # Targeted move: pick an element involved in a conflict
                conflict_elems = get_conflict_elements(curr_counts, curr_rows)
                if conflict_elems:
                    r_idx, j_idx = random.choice(conflict_elems)
                else:
                    r_idx = random.randint(0, n_rows - 1)
                    j_idx = random.randint(1, k)
            else:
                # Random move
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
            
            row = curr_rows[r_idx]
            old_val = row[j_idx]
            
            # Determine valid range for new value
            low = row[j_idx - 1] + 1
            # Allow expansion slightly beyond current M to escape local minima
            if j_idx < k:
                high = row[j_idx + 1] - 1
            else:
                high = curr_M + 15
            
            if high < low:
                continue
                
            new_val = random.randint(low, high)
            
            if new_val == old_val:
                continue
            
            # Remove old diffs and update penalty incrementally
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            
            # Tentatively set new value
            row[j_idx] = new_val
            
            # Add new diffs and update penalty incrementally
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            # Calculate new M using tracked row maxes
            old_row_max = row_maxes[r_idx]
            new_row_max = max(row)
            if new_row_max != old_row_max:
                row_maxes[r_idx] = new_row_max
            
            # Recompute M only if this row was the max or became the max
            if old_row_max == curr_M or new_row_max > curr_M:
                new_M = max(row_maxes)
            else:
                new_M = curr_M
            
            new_energy = compute_energy(new_M, curr_penalty, T)
            
            delta_E = new_energy - curr_energy
            
            # Acceptance criterion
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                curr_energy = new_energy
                curr_M = new_M
                
                if curr_penalty == 0:
                    if curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
            else:
                # Revert
                row[j_idx] = old_val
                row_maxes[r_idx] = old_row_max
                
                # Remove new diffs (which were added)
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                        
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                
                # Add old diffs (which were removed)
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                    
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1

                # Recalculate M if needed
                if old_row_max == curr_M:
                    curr_M = max(row_maxes)

            # Cool down
            T *= cooling
            if T < 0.5:
                T = 0.5

    # Final polish pass: try to reduce M by 1 repeatedly
    if best_rows is not None and best_M > 0:
        polish_deadline = deadline - 2
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            # Try to reduce each element that equals best_M
            for r_idx in range(n_rows):
                if time.time() > polish_deadline:
                    break
                row = best_rows[r_idx]
                for j_idx in range(k, 0, -1):
                    if time.time() > polish_deadline:
                        break
                    if row[j_idx] < best_M:
                        continue
                    # Try to reduce this element
                    old_val = row[j_idx]
                    low = row[j_idx - 1] + 1
                    high = old_val - 1
                    if high < low:
                        continue
                    # Try values from high down to low
                    for new_val in range(high, low - 1, -1):
                        # Check if this creates any conflicts
                        valid = True
                        for i in range(j_idx):
                            d = new_val - row[i]
                            # Count how many times this diff appears elsewhere
                            count = 0
                            for r2 in best_rows:
                                if r2 is row:
                                    continue
                                for a in range(len(r2)):
                                    for b in range(a+1, len(r2)):
                                        if r2[b] - r2[a] == d:
                                            count += 1
                            if count > 0:
                                valid = False
                                break
                            if not valid:
                                break
                        if valid:
                            for h in range(j_idx + 1, k + 1):
                                d = row[h] - new_val
                                count = 0
                                for r2 in best_rows:
                                    if r2 is row:
                                        continue
                                    for a in range(len(r2)):
                                        for b in range(a+1, len(r2)):
                                            if r2[b] - r2[a] == d:
                                                count += 1
                                if count > 0:
                                    valid = False
                                    break
                                if not valid:
                                    break
                        if valid:
                            row[j_idx] = new_val
                            best_M = max(best_M, max(max(r) for r in best_rows))
                            # Recompute actual M
                            actual_M = max(max(r) for r in best_rows)
                            if actual_M < best_M:
                                best_M = actual_M
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    if best_rows is None:
        best_rows = greedy_init()
        
    return best_rows
