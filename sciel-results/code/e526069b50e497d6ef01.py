import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
        
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    def calc_energy(rows, diffs):
        max_el = max(max(r) for r in rows)
        conflicts = 0
        for i in range(n_rows):
            for j in range(i+1, n_rows):
                conflicts += len(diffs[i] & diffs[j])
        return max_el * 1000 + conflicts

    def simulated_annealing(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_diffs = [get_diffs(r) for r in current_rows]
        current_energy = calc_energy(current_rows, current_diffs)
        
        best_rows = [r[:] for r in current_rows]
        best_energy = current_energy
        best_max = max(max(r) for r in best_rows)
        if best_energy % 1000 != 0:
             # Initial might be invalid? Greedy is valid.
             best_max = max(max(r) for r in best_rows)
        
        T = 5000.0
        T_min = 10.0
        cooling = 0.9995
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < T_min:
                T = T_min
                
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            
            low = current_rows[r_idx][j_idx - 1] + 1
            if j_idx < k:
                high = current_rows[r_idx][j_idx + 1] - 1
                if low > high:
                    continue
                new_val = random.randint(low, high)
            else:
                old_val = current_rows[r_idx][j_idx]
                high = max(low, old_val + 50)
                new_val = random.randint(low, high)
                
            new_row = current_rows[r_idx][:]
            new_row[j_idx] = new_val
            
            if not all(new_row[i] < new_row[i+1] for i in range(k)):
                continue
                
            new_diffs_row = get_diffs(new_row)
            
            temp_rows = current_rows[:]
            temp_rows[r_idx] = new_row
            temp_diffs = current_diffs[:]
            temp_diffs[r_idx] = new_diffs_row
            
            new_energy = calc_energy(temp_rows, temp_diffs)
            
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                current_rows = temp_rows
                current_diffs = temp_diffs
                current_energy = new_energy
                
                if new_energy % 1000 == 0:
                    cur_max = new_energy // 1000
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [r[:] for r in current_rows]
                        
            T *= cooling
            
        return best_rows, best_max

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_max = max(max(r) for r in best_rows)
    
    time_limit = 100.0
    final_rows, final_max = simulated_annealing(current_rows, time_limit)
    
    if final_max < best_max:
        best_rows = final_rows
        best_max = final_max
        
    return best_rows
