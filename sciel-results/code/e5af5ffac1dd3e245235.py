import random
import time
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_state_info(rows):
        diff_counter = Counter()
        max_span = 0
        for row in rows:
            max_span = max(max_span, row[-1])
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    diff_counter[diff] += 1
        duplicates = sum(c - 1 for c in diff_counter.values() if c > 1)
        return diff_counter, duplicates, max_span

    def run_sa(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        diff_counter, duplicates, max_span = get_state_info(rows)
        
        best_valid_rows = [r[:] for r in rows]
        best_valid_cost = max_span if duplicates == 0 else float('inf')
        current_cost = duplicates * 1000 + max_span
        
        temperature = 100.0
        cooling = 0.999999
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = random.randint(0, n_rows - 1)
            row = rows[i]
            if len(row) <= 1:
                continue
            j = random.randint(1, len(row) - 1)
            
            old_val = row[j]
            low = row[j-1] + 1
            high = row[j+1] - 1 if j < len(row) - 1 else old_val + 50
            
            if low > high:
                continue
                
            new_val = random.randint(low, high)
            if new_val == old_val:
                continue
            
            old_diffs = []
            new_diffs = []
            for l in range(len(row)):
                if l == j: continue
                old_diffs.append(abs(old_val - row[l]))
                new_diffs.append(abs(new_val - row[l]))
                
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] += 1
                
            new_duplicates = sum(c - 1 for c in diff_counter.values() if c > 1)
            new_max_span = max(max(r) for r in rows)
            
            new_cost = new_duplicates * 1000 + new_max_span
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / temperature):
                row[j] = new_val
                current_cost = new_cost
                max_span = new_max_span
                duplicates = new_duplicates
                
                if duplicates == 0 and max_span < best_valid_cost:
                    best_valid_cost = max_span
                    best_valid_rows = [r[:] for r in rows]
            else:
                for d in old_diffs:
                    diff_counter[d] += 1
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                        
            temperature *= cooling
            
        return best_valid_rows, best_valid_cost

    # Generate initial solutions
    best_overall_rows = None
    best_overall_cost = float('inf')
    
    # Run 5 restarts
    num_restarts = 5
    time_per_run = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 1: break
        
        # Generate initial solution
        # Use randomized greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # With some probability, try a larger x to diversify
                if random.random() < 0.2:
                    x += random.randint(0, 5)
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
            
        res_rows, res_cost = run_sa(rows, time_per_run)
        
        if res_cost < best_overall_cost:
            best_overall_cost = res_cost
            best_overall_rows = res_rows
            
    if best_overall_rows is None:
        # Fallback to greedy
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
        
    return best_overall_rows
