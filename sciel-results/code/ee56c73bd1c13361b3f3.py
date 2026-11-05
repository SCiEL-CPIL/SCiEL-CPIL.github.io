import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Total number of differences per row is k*(k+1)/2
    # Total differences in the set is n_rows * k*(k+1)/2
    # We need these to be distinct.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in get_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def greedy_random_row(used_diffs, target_max, randomize=True):
        """
        Construct a single row of length k+1 starting at 0.
        Tries to keep values below target_max.
        """
        row = [0]
        # We need k more elements
        for _ in range(k):
            last = row[-1]
            found = False
            # Randomize start search
            if randomize:
                # Try a random offset from last
                # Or just iterate
                pass
            
            # Search for next x
            # To ensure we don't loop forever, we have a target_max
            # If we exceed target_max, we might fail or return a larger value
            x = last + 1
            max_x = target_max + 50 # Allow some overflow
            
            # Collect valid candidates
            candidates = []
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    candidates.append(x)
                    if not randomize:
                        break
                    if len(candidates) > 10:
                        break
                x += 1
            
            if not candidates:
                # Failed to find a candidate within max_x
                # Try extending max_x? Or return None
                # For robustness, let's extend a bit more if randomize is false
                if not randomize:
                     while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            candidates.append(x)
                            break
                        x += 1
                        if x > 500: return None
                else:
                    return None
            
            if randomize:
                # Pick a candidate. Bias towards smaller?
                # Uniform among candidates found so far
                x = random.choice(candidates)
            else:
                x = candidates[0]
                
            row.append(x)
            used_diffs.update({x - a for a in row[:-1]})
            
        return row

    def initial_solution():
        """
        Create an initial set of rows.
        """
        used = set()
        rows = []
        # Start with a loose bound
        current_bound = 200
        for i in range(n_rows):
            row = greedy_random_row(used, current_bound, randomize=True)
            if row is None:
                # Retry with larger bound
                row = greedy_random_row(used, current_bound + 100, randomize=False)
                if row is None:
                    return None
            rows.append(row)
            # Update used diffs
            for d in get_diffs(row):
                used.add(d)
            # Tighten bound slightly for next row? Or keep it loose?
            # Let's keep it loose to ensure completion, then optimize.
        return rows

    def calculate_energy(rows):
        """
        Energy function: Max value + small penalty for sum of squares.
        """
        max_val = max(row[-1] for row in rows)
        sum_sq = 0
        for row in rows:
            for v in row:
                sum_sq += v * v
        # Weight for sum_sq should be small so max_val dominates
        return max_val + 0.01 * sum_sq

    def try_move(rows, used_diffs, temp):
        """
        Attempt a random move. Returns (new_rows, new_energy, accepted).
        """
        # Choose a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Choose a random index to modify (1 to k, 0 is fixed)
        i_idx = random.randint(1, k)
        
        # Choose a delta
        # Mostly small deltas, occasionally larger
        if random.random() < 0.7:
            delta = random.randint(1, 3)
        else:
            delta = random.randint(1, 10)
            
        old_val = row[i_idx]
        new_val = old_val - delta
        
        # Check bounds
        if new_val <= row[i_idx-1]:
            return None, None, False
            
        # Check if new_val creates valid diffs
        # We need to check diffs within the row and against other rows
        # Efficient check:
        # 1. Diffs involving new_val within the row
        # 2. Diffs not involving new_val within the row remain same
        # 3. Diffs in other rows remain same
        
        # Let's just reconstruct the row's diffs and check intersection
        # This is O(k^2) per move, which is fine for k=7
        
        new_row = row.copy()
        new_row[i_idx] = new_val
        
        # Check strict increase
        # Already checked new_val > row[i_idx-1]
        # Need to check new_val < row[i_idx+1] if i_idx < k
        if i_idx < k and new_val >= row[i_idx+1]:
             return None, None, False
             
        # Calculate new diffs for this row
        new_row_diffs = get_diffs(new_row)
        
        # Calculate diffs for other rows
        other_diffs = set()
        for r in range(n_rows):
            if r == r_idx:
                continue
            other_diffs.update(get_diffs(rows[r]))
            
        # Check intersection
        if new_row_diffs & other_diffs:
            return None, None, False
            
        # If we reach here, the move is valid
        # Calculate new energy
        new_rows = [r.copy() if r != r_idx else new_row for r in rows]
        new_energy = calculate_energy(new_rows)
        
        old_energy = calculate_energy(rows)
        delta_e = new_energy - old_energy
        
        # Acceptance criterion
        if delta_e < 0:
            return new_rows, new_energy, True
        else:
            if temp > 0:
                prob = math.exp(-delta_e / temp)
                if random.random() < prob:
                    return new_rows, new_energy, True
        return None, None, False

    def regenerate_row(rows, used_diffs_except_row, r_idx, target_max):
        """
        Regenerate row r_idx completely.
        """
        # Remove diffs of row r_idx from used set? 
        # The function signature implies we pass the set of diffs from OTHER rows.
        new_row = greedy_random_row(used_diffs_except_row, target_max, randomize=True)
        if new_row is None:
             # Try deterministic with higher bound
             new_row = greedy_random_row(used_diffs_except_row, target_max + 100, randomize=False)
             if new_row is None:
                 return None
        return new_row

    # Main SA Loop
    best_solution = None
    best_energy = float('inf')
    best_max = float('inf')
    
    # Initialize
    rows = initial_solution()
    if rows is None:
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
             row = greedy_random_row(used, 300, randomize=False)
             if row is None: return [list(range(k+1))] # Should not happen
             rows.append(row)
             used.update(get_diffs(row))
        return rows
        
    current_energy = calculate_energy(rows)
    current_max = max(row[-1] for row in rows)
    best_solution = [r.copy() for r in rows]
    best_energy = current_energy
    best_max = current_max
    
    # Pre-calculate diffs for each row to speed up moves? 
    # For k=7, get_diffs is fast enough.
    
    # Annealing parameters
    t = 10.0
    cooling = 0.995
    t_min = 0.1
    
    while time.time() - start_time < time_limit and t > t_min:
        # Determine move type
        move_type = random.random()
        
        if move_type < 0.8:
            # Local shift
            new_rows, new_energy, accepted = try_move(rows, None, t)
            if accepted:
                rows = new_rows
                current_energy = new_energy
                cur_max = max(row[-1] for row in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_energy = current_energy
                    best_solution = [r.copy() for r in rows]
        else:
            # Regenerate one row
            r_idx = random.randint(0, n_rows - 1)
            # Get diffs from other rows
            other_diffs = set()
            for r in range(n_rows):
                if r == r_idx:
                    continue
                other_diffs.update(get_diffs(rows[r]))
            
            # Target max: try to keep it around current max or slightly lower
            target_max = current_max - random.randint(0, 10)
            if target_max < 10: target_max = 10
            
            new_row = regenerate_row(rows, other_diffs, r_idx, target_max)
            if new_row is not None:
                # Check if this new row is valid with others (regenerate should ensure this, but double check just in case)
                new_rows = [r.copy() if r != r_idx else new_row for r in rows]
                # Verify validity
                valid = True
                all_diffs = set()
                for r in new_rows:
                    rd = get_diffs(r)
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs.update(rd)
                
                if valid:
                    new_energy = calculate_energy(new_rows)
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or (t > 0 and random.random() < math.exp(-delta_e / t)):
                        rows = new_rows
                        current_energy = new_energy
                        cur_max = max(row[-1] for row in rows)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_energy = current_energy
                            best_solution = [r.copy() for r in rows]
        
        t *= cooling
        
        # If we hit the theoretical lower bound, we can stop?
        # Lower bound approx: n_rows * k * (k+1) / 2 is total diffs.
        # Max diff M must be at least total_diffs.
        # For n=4, k=7: 4 * 28 = 112 diffs. So M >= 112.
        # If best_max <= 112, we found the optimal (or close).
        if best_max <= 112:
            break

    return best_solution
