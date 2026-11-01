import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Problem parameters
    size = k + 1
    # Pre-compute pairs (i, j) with i < j for difference calculation
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    n_pairs = len(pairs)
    total_diffs = n_rows * n_pairs

    # Helper to get all differences from a set of rows
    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            for i, j in pairs:
                diffs.append(r[j] - r[i])
        return diffs

    # Helper to get scope (max element)
    def get_scope(rows):
        return max(max(r) for r in rows)

    # Helper to calculate penalty: Scope + 1000 * Conflicts
    # Conflicts = total_diffs - number_of_unique_diffs
    def calc_penalty(rows):
        scope = get_scope(rows)
        all_diffs = get_all_diffs(rows)
        unique_diffs = len(set(all_diffs))
        conflicts = total_diffs - unique_diffs
        # We prioritize validity heavily. If valid, minimize scope.
        # If invalid, minimize conflicts first (by making penalty huge).
        return scope + 1000 * conflicts

    # Initial construction: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < size:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_penalty = calc_penalty(best_rows)
    best_scope = get_scope(best_rows)
    
    # Ensure we have a valid baseline if greedy failed (unlikely but possible)
    if best_penalty > 1000:
        # Fallback: simple random restart greedy
        for _ in range(10):
            rows = greedy_construct()
            p = calc_penalty(rows)
            if p < best_penalty:
                best_penalty = p
                best_rows = rows
                best_scope = get_scope(rows)
            if best_penalty < 1000:
                break

    # Simulated Annealing / Iterated Local Search
    current_rows = [r[:] for r in best_rows]
    current_penalty = best_penalty
    
    # Parameters
    T = 50.0  # Initial temperature
    cooling = 0.999
    min_T = 0.1
    steps_per_iter = 5000
    max_restarts = 20
    
    # Pre-compute neighbors for fast move generation?
    # No, dynamic range is better.
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Reset current state to best known, with a small kick if not first restart
        if restart > 0:
            current_rows = [r[:] for r in best_rows]
            current_penalty = best_penalty
            # Apply a kick to escape local minima
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, size - 1)
                lo = current_rows[r_idx][pos - 1] + 1
                hi = current_rows[r_idx][pos + 1] - 1 if pos < size - 1 else current_rows[r_idx][pos] + 20
                if lo <= hi:
                    current_rows[r_idx][pos] = random.randint(lo, hi)
            current_penalty = calc_penalty(current_rows)
            
        T = 50.0
        no_improve_count = 0
        local_best_penalty = current_penalty
        local_best_rows = [r[:] for r in current_rows]

        for step in range(steps_per_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random move
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, size - 1)
            
            old_val = current_rows[r_idx][pos]
            
            # Determine valid range for new value
            lo = current_rows[r_idx][pos - 1] + 1
            hi = current_rows[r_idx][pos + 1] - 1 if pos < size - 1 else old_val + 10
            
            if lo > hi:
                # Try to expand hi if it's the last element
                if pos == size - 1:
                    hi = old_val + 10
                else:
                    continue
                    
            # Generate new value
            # Mix of small steps and large jumps
            if random.random() < 0.1:
                new_val = random.randint(lo, hi)
            else:
                # Small step
                max_step = 5
                delta = random.randint(-max_step, max_step)
                new_val = old_val + delta
                if new_val < lo: new_val = lo
                if new_val > hi: new_val = hi
            
            if new_val == old_val:
                continue

            # Apply move
            current_rows[r_idx][pos] = new_val
            
            # Check strict increasing (sanity check)
            if (pos > 0 and current_rows[r_idx][pos] <= current_rows[r_idx][pos - 1]) or \
               (pos < size - 1 and current_rows[r_idx][pos] >= current_rows[r_idx][pos + 1]):
                current_rows[r_idx][pos] = old_val
                continue

            new_penalty = calc_penalty(current_rows)
            delta_penalty = new_penalty - current_penalty
            
            # Acceptance criterion
            if delta_penalty < 0 or random.random() < math.exp(-delta_penalty / T):
                current_penalty = new_penalty
                
                if new_penalty < local_best_penalty:
                    local_best_penalty = new_penalty
                    local_best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                current_rows[r_idx][pos] = old_val
            
            T *= cooling
            if T < min_T:
                T = min_T
                
            # Early termination for this iteration if stuck and cold
            if no_improve_count > 1000 and T < 1.0:
                break

        # Update global best if local search found something better
        if local_best_penalty < best_penalty:
            best_penalty = local_best_penalty
            best_rows = local_best_rows
            best_scope = get_scope(best_rows)

    # Final polish: Try to reduce scope further with a tight local search on the best valid solution
    # If best_solution is valid, we can try to minimize scope specifically.
    if best_penalty < 1000: # It is valid
        # Use a simple hill climbing on scope only, ensuring validity is maintained
        # Actually, the SA already minimized scope + conflicts. 
        # Since conflicts=0, it minimized scope.
        # But sometimes SA gets stuck on a valid solution with slightly higher scope than possible.
        # Let's do a few iterations of "try to reduce max element"
        
        current_rows = [r[:] for r in best_rows]
        current_scope = get_scope(current_rows)
        
        for _ in range(10000):
            if time.time() > deadline - 1.0:
                break
            improved = False
            # Try to reduce the largest elements
            # Find the row with the max element
            max_val = current_scope
            # Try to move any element that is max_val to a smaller value
            # Or just random search for scope reduction
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, size - 1)
            old_val = current_rows[r_idx][pos]
            
            # Try to reduce value
            lo = current_rows[r_idx][pos - 1] + 1
            hi = old_val - 1
            if pos < size - 1:
                hi = min(hi, current_rows[r_idx][pos + 1] - 1)
            
            if lo <= hi:
                # Try a few random values in [lo, hi]
                for _ in range(5):
                    new_val = random.randint(lo, hi)
                    if new_val == old_val:
                        continue
                    current_rows[r_idx][pos] = new_val
                    # Check validity
                    all_diffs = get_all_diffs(current_rows)
                    if len(set(all_diffs)) == total_diffs:
                        new_scope = get_scope(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                            break
                    # Revert if not better or invalid
                    current_rows[r_idx][pos] = old_val
                    if improved:
                        break
            
            if not improved:
                # No improvement in this step, maybe restart this phase?
                pass

        if get_scope(current_rows) < best_scope:
            best_rows = current_rows
            best_scope = get_scope(best_rows)

    return best_rows
