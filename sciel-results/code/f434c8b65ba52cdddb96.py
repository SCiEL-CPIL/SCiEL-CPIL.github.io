import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Total number of differences
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Estimate a starting bound
    initial_max_val = int(total_diffs * 1.5) + 50

    def get_energy(rows):
        max_val = 0
        diffs = []
        for row in rows:
            if row[-1] > max_val:
                max_val = row[-1]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        
        unique_diffs = set(diffs)
        conflicts = len(diffs) - len(unique_diffs)
        
        # Weight for conflicts must be high enough to dominate max_val
        # max_val is ~100-200. conflicts can be up to total_diffs.
        # 1000 is a safe multiplier.
        return max_val + 1000 * conflicts

    def generate_random_rows(max_val):
        rows = []
        for _ in range(n_rows):
            available = list(range(1, max_val + 1))
            if len(available) < k:
                row = list(range(0, k + 1))
            else:
                selected = random.sample(available, k)
                selected.sort()
                row = [0] + selected
            rows.append(row)
        return rows

    def valid_move(rows, r_idx, j, new_val):
        row = rows[r_idx]
        if j == 0:
            return new_val == 0
        if new_val <= row[j-1]:
            return False
        if j < len(row) - 1 and new_val >= row[j+1]:
            return False
        return True

    def kick(rows, max_val_limit):
        """
        Randomize one row to escape local minima.
        """
        r_idx = random.randint(0, n_rows - 1)
        # Regenerate this row
        available = list(range(1, max_val_limit + 1))
        if len(available) < k:
            new_row = list(range(0, k + 1))
        else:
            selected = random.sample(available, k)
            selected.sort()
            new_row = [0] + selected
        rows[r_idx] = new_row

    # Simulated Annealing
    current_rows = generate_random_rows(initial_max_val)
    current_energy = get_energy(current_rows)
    
    best_rows = [r.copy() for r in current_rows]
    best_energy = current_energy
    
    T = 100.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    no_improve_count = 0
    kick_interval = 500  # Kick every 500 iterations if no improvement
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[r_idx][j]
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        if valid_move(current_rows, r_idx, j, new_val):
            current_rows[r_idx][j] = new_val
            new_energy = get_energy(current_rows)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_rows = [r.copy() for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                current_rows[r_idx][j] = old_val
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Kick if stuck
        if no_improve_count > kick_interval:
            # Use current max val as limit for kick to avoid expanding too much
            current_max = max(row[-1] for row in current_rows)
            kick(current_rows, current_max)
            current_energy = get_energy(current_rows)
            if current_energy < best_energy:
                best_energy = current_energy
                best_rows = [r.copy() for r in current_rows]
            no_improve_count = 0

        # Cool down
        T *= cooling_rate

    return best_rows
