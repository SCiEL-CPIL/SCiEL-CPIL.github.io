"""Improved difference triangle set construction using randomized greedy + local search."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def all_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def valid(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for d in all_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Phase 1: Multiple randomized greedy constructions
    best_rows = None
    best_score = float('inf')
    
    num_greedy = 50
    start_time = time.time()
    
    for trial in range(num_greedy):
        if time.time() > deadline - 60:
            break
        
        random.seed(trial * 7919 + 13)
        
        used = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                # With some probability, skip a valid position to add diversity
                if random.random() < 0.15 and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        if valid(rows):
            score = max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = [row[:] for row in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Phase 2: Local search / hill climbing to reduce max entry
    if time.time() < deadline - 10:
        current = [row[:] for row in best_rows]
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Try to improve by adjusting individual entries
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random row and a random position to try moving
                r = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)  # don't move the 0
                
                old_val = current[r][pos]
                
                # Try to decrease the value
                new_val = max(1, old_val - random.randint(1, 3))
                
                # Check if new_val maintains increasing order
                if pos > 0 and new_val <= current[r][pos - 1]:
                    continue
                if pos < k and new_val >= current[r][pos + 1]:
                    continue
                
                # Try the change
                current[r][pos] = new_val
                
                if valid(current):
                    new_score = max_entry(current)
                    if new_score < current_score:
                        current_score = new_score
                        best_rows = [row[:] for row in current]
                        improved = True
                        no_improve_count = 0
                    else:
                        # Accept with some probability (simulated annealing style)
                        if random.random() < 0.1:
                            pass  # keep the change
                        else:
                            current[r][pos] = old_val
                else:
                    current[r][pos] = old_val
            
            if not improved:
                no_improve_count += 1
    
    return best_rows
