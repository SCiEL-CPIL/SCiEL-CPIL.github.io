import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    total_diffs = n_rows * k * (k + 1) // 2
    
    def make_diffs_matrix(rows):
        """Compute difference matrix from rows. rows is list of lists or 2D array."""
        rows_arr = np.array(rows, dtype=np.int64)
        n = rows_arr.shape[1]
        diffs = np.zeros((n_rows, k + 1, k + 1), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                diffs[:, i, j] = rows_arr[:, j] - rows_arr[:, i]
        return diffs
    
    def is_valid(rows):
        if rows is None:
            return False
        rows_arr = np.array(rows, dtype=np.int64)
        if rows_arr.shape != (n_rows, k + 1):
            return False
        # Check increasing
        for r in range(n_rows):
            if rows_arr[r, 0] != 0:
                return False
            if np.any(np.diff(rows_arr[r]) <= 0):
                return False
        # Check all positive differences distinct
        diffs = make_diffs_matrix(rows)
        pos_mask = np.triu(np.ones((k + 1, k + 1), dtype=bool), 1)
        pos_diffs = diffs[:, pos_mask[0, :].nonzero()[0], pos_mask[0, :].nonzero()[1]]
        # Flatten and check uniqueness
        flat = pos_diffs.flatten()
        if len(np.unique(flat)) != len(flat):
            return False
        return True
    
    def get_score(rows):
        if rows is None:
            return float('inf')
        return int(max(max(row) for row in rows))
    
    def global_greedy(seed_val):
        """Vectorized global greedy: fill all rows simultaneously."""
        rng = random.Random(seed_val)
        rows = np.zeros((n_rows, k + 1), dtype=np.int64)
        # Row 0 is always 0
        rows[:, 0] = 0
        used = set()
        
        for col in range(1, k + 1):
            # For each row, find the smallest candidate that doesn't conflict
            # Candidates must be > previous element in that row
            prev = rows[:, col - 1]
            candidates = np.zeros(n_rows, dtype=np.int64)
            
            for r in range(n_rows):
                x = prev[r] + 1
                while True:
                    new_diffs = [x - rows[r, i] for i in range(col)]
                    if not any(d in used for d in new_diffs):
                        candidates[r] = x
                        break
                    x += 1
                    if x > 500:
                        return None
            
            # Update used set
            for r in range(n_rows):
                x = candidates[r]
                for i in range(col):
                    d = x - rows[r, i]
                    used.add(d)
                rows[r, col] = x
        
        # Verify
        if not is_valid(rows.tolist()):
            return None
        return rows.tolist()
    
    def global_greedy_randomized(seed_val, noise_level=0.0):
        """Global greedy with random noise to escape local minima."""
        rng = random.Random(seed_val)
        rows = np.zeros((n_rows, k + 1), dtype=np.int64)
        rows[:, 0] = 0
        used = set()
        
        for col in range(1, k + 1):
            prev = rows[:, col - 1]
            candidates = np.zeros(n_rows, dtype=np.int64)
            
            for r in range(n_rows):
                x = prev[r] + 1
                # With some probability, skip a few candidates
                skip = 0
                if noise_level > 0:
                    skip = rng.randint(0, int(noise_level * 5))
                x += skip
                while True:
                    new_diffs = [x - rows[r, i] for i in range(col)]
                    if not any(d in used for d in new_diffs):
                        candidates[r] = x
                        break
                    x += 1
                    if x > 500:
                        return None
            
            for r in range(n_rows):
                x = candidates[r]
                for i in range(col):
                    d = x - rows[r, i]
                    used.add(d)
                rows[r, col] = x
        
        if not is_valid(rows.tolist()):
            return None
        return rows.tolist()
    
    def simulated_annealing(rows, seed_val, max_iter=5000):
        """Simulated annealing on a valid solution to reduce max entry."""
        rng = random.Random(seed_val)
        rows = [row[:] for row in rows]
        current_score = get_score(rows)
        best_score = current_score
        best_rows = [row[:] for row in rows]
        
        # Precompute all differences for quick validity checking
        def compute_used_diffs(row_list):
            used = set()
            for row in row_list:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        def try_move(rows, move_type, r, pos, val):
            """Try a move and return (new_rows, delta_score, is_valid)"""
            old_val = rows[r][pos]
            if val <= rows[r][pos - 1] or val >= rows[r][pos + 1] if pos < k else val <= rows[r][pos - 1]:
                return None, 0, False
            if pos == 0:
                return None, 0, False
            
            # Remove old differences involving position pos
            old_diffs = set()
            for i in range(len(rows[r])):
                if i != pos:
                    if i < pos:
                        old_diffs.add(rows[r][pos] - rows[r][i])
                    else:
                        old_diffs.add(rows[r][i] - rows[r][pos])
            
            # Check new value doesn't conflict
            rows[r][pos] = val
            new_diffs = set()
            for i in range(len(rows[r])):
                if i != pos:
                    if i < pos:
                        new_diffs.add(val - rows[r][i])
                    else:
                        new_diffs.add(rows[r][i] - val)
            
            # Check new diffs don't conflict with existing (excluding old diffs)
            # Build set of all diffs except those from row r
            other_diffs = set()
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                orow = rows[other_r]
                for i in range(len(orow)):
                    for j in range(i + 1, len(orow)):
                        other_diffs.add(orow[j] - orow[i])
            
            # Also need to check within row r excluding pos
            row_r_diffs = set()
            for i in range(len(rows[r])):
                for j in range(i + 1, len(rows[r])):
                    if i != pos and j != pos:
                        row_r_diffs.add(rows[r][j] - rows[r][i])
            
            all_existing = other_diffs | row_r_diffs
            if new_diffs & all_existing:
                rows[r][pos] = old_val
                return None, 0, False
            
            # Check internal consistency of row r
            row = rows[r]
            if pos < k and val >= row[pos + 1]:
                rows[r][pos] = old_val
                return None, 0, False
            if pos > 0 and val <= row[pos - 1]:
                rows[r][pos] = old_val
                return None, 0, False
            
            new_score = get_score(rows)
            delta = new_score - current_score
            return rows, delta, True
        
        temperature = 5.0
        cooling = 0.9995
        iterations = 0
        
        while iterations < max_iter and temperature > 0.01:
            if time.time() > deadline - 5:
                break
            
            # Choose random move
            r = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1 to k
            move_type = rng.choice(['shift_down', 'shift_up', 'replace'])
            
            old_val = rows[r][pos]
            
            if move_type == 'shift_down':
                # Try to decrease
                delta_val = -rng.randint(1, 3)
                new_val = old_val + delta_val
                if new_val <= rows[r][pos - 1]:
                    continue
            elif move_type == 'shift_up':
                delta_val = rng.randint(1, 2)
                new_val = old_val + delta_val
                if pos < k and new_val >= rows[r][pos + 1]:
                    continue
            else:  # replace
                # Try a random value in range
                lo = rows[r][pos - 1] + 1
                hi = rows[r][pos + 1] - 1 if pos < k else old_val + 5
                if lo >= hi:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
            
            # Apply move
            rows[r][pos] = new_val
            
            # Quick validity check
            row = rows[r]
            valid_move = True
            if pos < k and new_val >= row[pos + 1]:
                valid_move = False
            if pos > 0 and new_val <= row[pos - 1]:
                valid_move = False
            
            if valid_move:
                # Check differences
                new_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                
                # Check against other rows
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    orow = rows[other_r]
                    for i in range(len(orow)):
                        for j in range(i + 1, len(orow)):
                            if orow[j] - orow[i] in new_diffs:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                
                if not conflict:
                    # Check within same row (excluding pos)
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if i != pos and j != pos:
                                if row[j] - row[i] in new_diffs:
                                    conflict = True
                                    break
                        if conflict:
                            break
                
                if conflict:
                    rows[r][pos] = old_val
                    continue
            
            # Accept/reject based on SA
            new_score = get_score(rows)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [row[:] for row in rows]
            else:
                rows[r][pos] = old_val
            
            temperature *= cooling
            iterations += 1
        
        return best_rows, best_score
    
    # Strategy 1: Global greedy (deterministic)
    best_rows = None
    best_score = float('inf')
    
    rows = global_greedy(42)
    if rows is not None:
        s = get_score(rows)
        if s < best_score:
            best_score = s
            best_rows = rows
    
    # Strategy 2: Multiple randomized global greedy attempts
    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline - 30:
            break
        seed = attempt * 7919 + 13
        noise = 0.02 + 0.08 * (attempt % 5) / 5.0
        rows = global_greedy_randomized(seed, noise_level=noise)
        if rows is not None:
            s = get_score(rows)
            if s < best_score:
                best_score = s
                best_rows = rows
    
    # Strategy 3: Simulated annealing on best solution
    if best_rows is not None and time.time() < deadline - 10:
        for sa_seed in range(10):
            if time.time() > deadline - 5:
                break
            improved_rows, improved_score = simulated_annealing(
                best_rows, sa_seed * 1000 + 7, max_iter=3000
            )
            if improved_score < best_score:
                best_score = improved_score
                best_rows = improved_rows
    
    # Strategy 4: More SA with different starting points
    if best_rows is not None and time.time() < deadline - 5:
        for extra_seed in range(5):
            if time.time() > deadline - 3:
                break
            # Create a perturbed version of best_rows
            perturbed = [row[:] for row in best_rows]
            rng = random.Random(50000 + extra_seed)
            for _ in range(3):
                r = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)
                if pos < k:
                    perturbed[r][pos] += rng.randint(-2, 2)
                else:
                    perturbed[r][pos] += rng.randint(-2, 0)
            # Validate perturbed
            if is_valid(perturbed):
                improved_rows, improved_score = simulated_annealing(
                    perturbed, 60000 + extra_seed * 100, max_iter=2000
                )
                if improved_score < best_score:
                    best_score = improved_score
                    best_rows = improved_rows
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: simple greedy
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
# EVOLVE-BLOCK-END
