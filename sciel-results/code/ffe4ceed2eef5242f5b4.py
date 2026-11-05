import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def compute_all_diffs(rows):
        all_diffs = {}
        for i, row in enumerate(rows):
            all_diffs[i] = get_row_diffs(row)
        return all_diffs
    
    def greedy_init():
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows
    
    best_solution = greedy_init()
    best_max = max(max(r) for r in best_solution)
    best_diffs = compute_all_diffs(best_solution)
    
    def try_improve(rows, diffs_dict, current_max, max_iters=20000):
        nonlocal best_max, best_solution, best_diffs
        
        rows = [list(r) for r in rows]
        diffs_dict = {i: set(d) for i, d in diffs_dict.items()}
        current_max = max(max(r) for r in rows)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            candidates = []
            for delta in range(-30, 31):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= rows[row_idx][pos-1]:
                    continue
                if pos < k and new_val >= rows[row_idx][pos+1]:
                    continue
                candidates.append(new_val)
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            for new_val in candidates:
                new_row = list(rows[row_idx])
                new_row[pos] = new_val
                
                if any(new_row[i] >= new_row[i+1] for i in range(k)):
                    continue
                
                new_row_diffs = get_row_diffs(new_row)
                
                if len(new_row_diffs) != k*(k+1)//2:
                    continue
                
                conflict = False
                for i in range(n_rows):
                    if i != row_idx:
                        if new_row_diffs & diffs_dict[i]:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                other_max = max(max(rows[i]) for i in range(n_rows) if i != row_idx) if n_rows > 1 else 0
                new_max = max(other_max, max(new_row))
                
                if new_max < current_max:
                    rows[row_idx] = new_row
                    diffs_dict[row_idx] = new_row_diffs
                    current_max = new_max
                    break
                elif new_max == current_max and random.random() < 0.05:
                    rows[row_idx] = new_row
                    diffs_dict[row_idx] = new_row_diffs
                    break
        
        if current_max < best_max:
            best_max = current_max
            best_solution = [list(r) for r in rows]
            best_diffs = {i: set(d) for i, d in diffs_dict.items()}
        
        return rows, diffs_dict, current_max
    
    best_solution, best_diffs, best_max = try_improve(best_solution, best_diffs, best_max, max_iters=50000)
    
    while time.time() - start_time < time_limit - 15:
        perturbed = [list(r) for r in best_solution]
        perturbed_diffs = {i: set(d) for i, d in best_diffs.items()}
        
        num_changes = random.randint(2, 5)
        for _ in range(num_changes):
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            old_val = perturbed[row_idx][pos]
            
            found = False
            for _ in range(100):
                delta = random.randint(-20, 20)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= perturbed[row_idx][pos-1]:
                    continue
                if pos < k and new_val >= perturbed[row_idx][pos+1]:
                    continue
                
                new_row = list(perturbed[row_idx])
                new_row[pos] = new_val
                
                if any(new_row[i] >= new_row[i+1] for i in range(k)):
                    continue
                
                new_row_diffs = get_row_diffs(new_row)
                if len(new_row_diffs) != k*(k+1)//2:
                    continue
                
                conflict = False
                for i in range(n_rows):
                    if i != row_idx:
                        if new_row_diffs & perturbed_diffs[i]:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                perturbed[row_idx] = new_row
                perturbed_diffs[row_idx] = new_row_diffs
                found = True
                break
        
        if not found:
            continue
        
        all_diffs = set()
        valid = True
        for i in range(n_rows):
            if perturbed_diffs[i] & all_diffs:
                valid = False
                break
            all_diffs |= perturbed_diffs[i]
        
        if not valid:
            continue
        
        pert_max = max(max(r) for r in perturbed)
        
        improved_rows, improved_diffs, improved_max = try_improve(
            perturbed, perturbed_diffs, pert_max, max_iters=10000
        )
        
        if improved_max < best_max:
            best_max = improved_max
            best_solution = improved_rows
            best_diffs = improved_diffs
    
    return best_solution
