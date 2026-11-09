import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # k=5, n_rows=7
    # Each row has k+1 = 6 elements, starting at 0
    # So each row is (0, a1, a2, a3, a4, a5) with 0 < a1 < a2 < a3 < a4 < a5
    # The differences are all a_j - a_i for i < j
    # We need all differences across all rows to be distinct
    
    # Strategy: 
    # 1. Precompute a good set of rows using a smarter approach
    # 2. Use local search to minimize the maximum entry
    
    # First, let's try a constructive approach
    # We know the best known is 112, counting bound is 105
    
    # Let's use a multi-start local search with a better initialization
    
    def generate_greedy(seed_offset=0):
        random.seed(42 + seed_offset)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs
    
    def get_max_scope(rows):
        return max(max(r) for r in rows)
    
    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs
    
    def try_move(rows, used_diffs, r_idx, pos, new_val):
        """Try to set rows[r_idx][pos] = new_val. Returns True if valid."""
        row = rows[r_idx]
        old_val = row[pos]
        if new_val == old_val:
            return True
        # Check bounds
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < k and new_val >= row[pos + 1]:
            return False
        
        old_involved = get_involved_diffs(row, pos, old_val)
        used_diffs.difference_update(old_involved)
        
        new_involved = get_involved_diffs(row, pos, new_val)
        
        if len(new_involved) != len(old_involved):
            used_diffs |= old_involved
            return False
        
        if new_involved & used_diffs:
            used_diffs |= old_involved
            return False
        
        # Valid
        used_diffs |= new_involved
        row[pos] = new_val
        return True
    
    def revert_move(rows, used_diffs, r_idx, pos, old_val):
        """Revert a move."""
        row = rows[r_idx]
        new_val = row[pos]
        new_involved = get_involved_diffs(row, pos, new_val)
        used_diffs.difference_update(new_involved)
        old_involved = get_involved_diffs(row, pos, old_val)
        used_diffs |= old_involved
        row[pos] = old_val
    
    # Multi-start hill climbing
    best_rows = None
    best_scope = float('inf')
    
    num_starts = 5
    time_per_start = (deadline - start_time - 5.0) / num_starts
    
    for start_i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        rows, used_diffs = generate_greedy(start_i)
        current_scope = get_max_scope(rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
        
        # Hill climbing: repeatedly try to reduce the maximum entry
        improved = True
        start_hc = time.time()
        while improved and (time.time() - start_hc) < time_per_start * 0.5:
            improved = False
            # Find the row with the maximum entry
            max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
            max_row = rows[max_row_idx]
            max_val = max(max_row)
            max_pos = max_row.index(max_val)
            
            # Try to reduce the maximum value
            for new_val in range(max_row[max_pos - 1] + 1, max_val):
                if try_move(rows, used_diffs, max_row_idx, max_pos, new_val):
                    new_scope = get_max_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        break
                    else:
                        revert_move(rows, used_diffs, max_row_idx, max_pos, max_val)
            
            # Also try reducing other large values
            if not improved:
                for r_idx in range(n_rows):
                    row = rows[r_idx]
                    # Try positions from the end
                    for pos in range(k, 0, -1):
                        old_val = row[pos]
                        # Try to reduce
                        low = row[pos - 1] + 1
                        for new_val in range(low, old_val):
                            if try_move(rows, used_diffs, r_idx, pos, new_val):
                                new_scope = get_max_scope(rows)
                                if new_scope < current_scope:
                                    current_scope = new_scope
                                    if current_scope < best_scope:
                                        best_scope = current_scope
                                        best_rows = [r[:] for r in rows]
                                    improved = True
                                    break
                                else:
                                    revert_move(rows, used_diffs, r_idx, pos, old_val)
                        if improved:
                            break
                    if improved:
                        break
        
        # Simulated annealing for the rest of the time
        if time.time() > deadline - 5.0:
            break
        
        T = 3.0
        cooling = 0.995
        sa_start = time.time()
        while time.time() < min(deadline - 5.0, start_hc + time_per_start):
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, k)
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < k else old_val + 20
            
            if old_val == max(row):
                if low < old_val:
                    new_val = random.randint(low, old_val - 1)
                else:
                    continue
            else:
                search_low = max(low, old_val - 8)
                search_high = min(high, old_val + 8)
                if search_low > search_high:
                    continue
                new_val = random.randint(search_low, search_high)
            
            if new_val == old_val:
                continue
            
            if try_move(rows, used_diffs, r_idx, pos, new_val):
                new_scope = get_max_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_scope - current_scope
                    if random.random() < math.exp(-delta / T):
                        current_scope = new_scope
                    else:
                        revert_move(rows, used_diffs, r_idx, pos, old_val)
            
            T *= cooling
            if T < 0.1:
                T = 0.1
    
    if best_rows is None:
        # Fallback: return greedy
        rows, _ = generate_greedy(0)
        best_rows = rows
    
    return best_rows
