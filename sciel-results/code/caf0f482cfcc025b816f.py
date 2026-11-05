import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Phase 1: Generate multiple greedy seeds with randomization
    def greedy_construct(n_rows, k, rng):
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
    
    # Phase 2: Local search to reduce maximum value
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def local_search(rows, n_rows, k, rng, time_limit=10.0):
        """Try to reduce the maximum value by swapping elements within rows."""
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        best_rows = [list(r) for r in current_rows]
        best_max = current_max
        
        start = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start < time_limit and no_improve_count < max_no_improve:
            # Pick a random row and a random position to modify
            row_idx = rng.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            pos = rng.randint(1, len(row) - 1)  # Don't change position 0 (always 0)
            
            old_val = row[pos]
            # Try to reduce this value
            new_val = rng.randint(1, old_val - 1) if old_val > 1 else old_val
            
            if new_val == old_val:
                no_improve_count += 1
                continue
            
            # Check if new_val maintains strict increase
            if pos > 0 and new_val <= row[pos-1]:
                no_improve_count += 1
                continue
            if pos < len(row) - 1 and new_val >= row[pos+1]:
                no_improve_count += 1
                continue
            
            # Test validity
            test_rows = [list(r) for r in current_rows]
            test_rows[row_idx][pos] = new_val
            
            if is_valid(test_rows):
                new_max = max(max(r) for r in test_rows)
                if new_max < best_max:
                    current_rows = test_rows
                    best_rows = [list(r) for r in test_rows]
                    best_max = new_max
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        return best_rows, best_max
    
    # Phase 3: Simulated annealing for larger moves
    def simulated_annealing(rows, n_rows, k, rng, time_limit=20.0):
        """SA that can move elements to new positions (not just reduce)."""
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        best_rows = [list(r) for r in current_rows]
        best_max = current_max
        
        start = time.time()
        temp = 50.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            # Pick a random row and position
            row_idx = rng.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            pos = rng.randint(1, len(row) - 1)
            
            old_val = row[pos]
            # Propose a new value (can be higher or lower)
            delta = rng.randint(-20, 20)
            new_val = old_val + delta
            
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos-1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos+1]:
                continue
            
            test_rows = [list(r) for r in current_rows]
            test_rows[row_idx][pos] = new_val
            
            if is_valid(test_rows):
                new_max = max(max(r) for r in test_rows)
                delta_max = new_max - current_max
                
                if delta_max < 0:
                    current_rows = test_rows
                    current_max = new_max
                    if new_max < best_max:
                        best_rows = [list(r) for r in test_rows]
                        best_max = new_max
                else:
                    # Accept with probability exp(-delta/temp)
                    if delta_max > 0 and rng.random() < (2.71828 ** (-delta_max / temp)):
                        current_rows = test_rows
                        current_max = new_max
            
            temp *= cooling
            if temp < 0.01:
                break
        
        return best_rows, best_max
    
    # Main algorithm
    rng = random.Random(42)
    
    # Generate multiple diverse seeds
    candidates = []
    for seed in range(30):
        if time.time() > deadline:
            break
        rng = random.Random(seed * 1000 + 12345)
        rows = greedy_construct(n_rows, k, rng)
        max_val = max(max(r) for r in rows)
        candidates.append((max_val, rows))
    
    # Sort by max value
    candidates.sort(key=lambda x: x[0])
    
    best_max = candidates[0][0] if candidates else float('inf')
    best_rows = candidates[0][1] if candidates else None
    
    # Local search on top candidates
    for max_val, rows in candidates[:5]:
        if time.time() > deadline:
            break
        rng = random.Random(9999)
        improved_rows, improved_max = local_search(rows, n_rows, k, rng, time_limit=8.0)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
    
    # Simulated annealing on best candidate
    if time.time() < deadline - 25 and best_rows is not None:
        rng = random.Random(7777)
        sa_rows, sa_max = simulated_annealing(best_rows, n_rows, k, rng, time_limit=20.0)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final local search on the best
    if time.time() < deadline - 5 and best_rows is not None:
        rng = random.Random(5555)
        final_rows, final_max = local_search(best_rows, n_rows, k, rng, time_limit=10.0)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = greedy_construct(n_rows, k, random.Random(0))
    
    return best_rows
