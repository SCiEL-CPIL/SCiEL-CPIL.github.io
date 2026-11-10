import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using a phased approach:
    1. Multiple randomized greedy initializations with lookahead.
    2. Intensive local search to minimize the maximum entry.
    3. Perturbation and re-optimization loop.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def greedy_initial(n_rows, k, rng, lookahead=20):
        """
        Greedy initialization with randomized choice among valid candidates.
        Uses a small lookahead window to avoid immediate dead-ends.
        """
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                candidates = []
                # Search in a window to find valid candidates
                # We don't want to search too far to keep numbers small
                limit = x + lookahead + int(r_idx * 5) 
                while x < limit:
                    # Check if x is valid given current row and used_diffs
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        candidates.append((x, new_diffs))
                    
                    x += 1
                    if time.time() > deadline - 20:
                        return None

                if not candidates:
                    # If no candidates in window, expand window
                    x = limit
                    continue

                # Choose a candidate
                # Strategy: Pick randomly from the first few valid candidates to keep numbers small
                # but introduce randomness to explore different structures
                top_n = min(3, len(candidates))
                chosen_idx = rng.randint(0, top_n - 1)
                chosen_val, chosen_diffs = candidates[chosen_idx]
                
                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                
                # Next search starts from chosen_val + 1
                x = chosen_val + 1

            rows.append(row)
        return rows

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]
        if current == new_val:
            return True, used_diffs

        # Check bounds
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None

        # Collect old differences involving position ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])

        # Remove old differences from used set
        temp_used = used_diffs - old_diffs

        # Check if new_val creates any new conflicts
        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)

        if not valid:
            return False, None

        new_used = temp_used | new_diffs
        return True, new_used

    def local_search_reduce_max(rows, n_rows, k, max_iters=5000):
        """
        Local search specifically designed to reduce the maximum entry.
        Focuses on the row with the largest last element.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        no_improve_count = 0
        max_no_improve = 200

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break

            if no_improve_count > max_no_improve:
                break

            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]
            row = rows[max_row_idx]
            
            found_improvement = False

            # Strategy 1: Try to move the last element of the max row to a smaller value
            ci = len(row) - 1
            for new_val in range(max_val - 1, row[ci - 1], -1):
                success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    new_score = max(r[-1] for r in rows)
                    if new_score < best_score:
                        best_score = new_score
                    found_improvement = True
                    break

            # Strategy 2: If last element couldn't be reduced, try to reduce other elements in the max row
            # to free up "space" or differences, potentially allowing the last element to be reduced later
            # or just reducing the overall spread.
            if not found_improvement:
                for ci in range(len(row) - 2, 0, -1):
                    current = row[ci]
                    # Try to move this element to a smaller value
                    # We need to be careful not to break the order
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                        if success:
                            rows[max_row_idx][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            found_improvement = True
                            break
                    if found_improvement:
                        break

            # Strategy 3: If no improvement in max row, try to reduce elements in other rows
            # This might free up differences that were blocking the max row.
            if not found_improvement:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    row = rows[ri]
                    # Try to reduce the last element of other rows first
                    for ci in range(len(row) - 1, 0, -1):
                        current = row[ci]
                        for new_val in range(current - 1, row[ci - 1], -1):
                            success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                            if success:
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                new_score = max(r[-1] for r in rows)
                                if new_score < best_score:
                                    best_score = new_score
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break

            if found_improvement:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return rows, best_score

    def perturb(rows, rng, strength=3):
        """
        Apply random perturbations to the rows to escape local minima.
        """
        perturbed = [row[:] for row in rows]
        n_rows = len(perturbed)
        for _ in range(strength):
            # Pick a random element to move
            ri = rng.randint(0, n_rows - 1)
            ci = rng.randint(1, len(perturbed[ri]) - 1)
            current = perturbed[ri][ci]
            
            # Try to move it to a random nearby value
            # We need to check validity, so we just pick a random offset
            # and rely on the local search to fix it, but we must ensure validity
            # A safer perturbation: swap two elements in the same row or different rows
            # Let's try swapping two elements from different rows
            
            ri2 = rng.randint(0, n_rows - 1)
            ci2 = rng.randint(1, len(perturbed[ri2]) - 1)
            
            v1 = perturbed[ri][ci]
            v2 = perturbed[ri2][ci2]
            
            # Check if swap is valid
            # This is complex to check fully, so we just do it and validate later
            # Or better: only swap if they are in the same "range" roughly
            if abs(v1 - v2) < 20:
                perturbed[ri][ci] = v2
                perturbed[ri2][ci2] = v1
                
                # Check if the row is still sorted
                if perturbed[ri] != sorted(perturbed[ri]):
                    # Revert if not sorted
                    perturbed[ri][ci] = v1
                    perturbed[ri2][ci2] = v2
                elif perturbed[ri2] != sorted(perturbed[ri2]):
                    perturbed[ri][ci] = v1
                    perturbed[ri2][ci2] = v2
                else:
                    # Check validity of the whole set
                    # This is expensive, so we might skip full validation here
                    # and let the local search handle it, but if it's invalid,
                    # local search might fail.
                    # Let's do a quick check
                    valid, _ = validate(perturbed)
                    if not valid:
                        perturbed[ri][ci] = v1
                        perturbed[ri2][ci2] = v2

        return perturbed

    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time_remaining() < 30:
            break
        
        rng = random.Random(42 + restart)
        rows = greedy_initial(n_rows, k, rng)
        if rows is None:
            continue

        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]
            
            # If we find a very good solution early, we can spend more time optimizing it
            # But for now, just continue to find better initializations

    if best_rows is None:
        # Fallback: Simple construction
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows

    # Phase 2: Local search on the best solution
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=5000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    # Phase 3: Perturbation and Re-optimization Loop
    # This is the key to escaping local minima
    for restart in range(20):
        if time_remaining() < 20:
            break
        
        rng = random.Random(100 + restart)
        perturbed_rows = perturb(best_rows, rng, strength=2 + restart % 3)
        
        # Validate perturbed rows
        valid, score = validate(perturbed_rows)
        if not valid:
            continue
            
        # Local search on perturbed rows
        improved_rows, improved_score = local_search_reduce_max(perturbed_rows, n_rows, k, max_iters=3000)
        
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    # Phase 4: Final Polish
    if time_remaining() > 10:
        improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=2000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    return best_rows
# EVOLVE-BLOCK-END
