import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows difference triangles (Golomb rulers) of length k+1.
    Minimizes the maximum entry.
    """
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 1. Greedy Initialization
    rows = []
    used_diffs = set()
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
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max
    
    # 2. Simulated Annealing
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    T = 5.0  # Initial temperature
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random mark index (1 to k)
        m_idx = random.randint(1, k)
        
        old_val = row[m_idx]
        
        # Propose new value
        # Bias towards decreasing values to minimize max
        if random.random() < 0.7:
            # Try to decrease
            low = row[m_idx - 1] + 1
            high = old_val - 1
            if low <= high:
                new_val = random.randint(low, high)
            else:
                new_val = old_val
        else:
            # Try to increase or stay (to escape local minima)
            low = old_val
            high = old_val + 5
            # Ensure it doesn't exceed next mark if it exists
            if m_idx < k:
                high = min(high, row[m_idx + 1] - 1)
            if low <= high:
                new_val = random.randint(low, high)
            else:
                new_val = old_val
                
        if new_val == old_val:
            continue
            
        # Check validity
        # 1. Order
        if m_idx > 0 and new_val <= row[m_idx - 1]:
            continue
        if m_idx < k and new_val >= row[m_idx + 1]:
            continue
            
        # 2. Differences
        # Old diffs involving this mark
        old_diffs_inv = set()
        for i in range(m_idx):
            old_diffs_inv.add(old_val - row[i])
        for i in range(m_idx + 1, len(row)):
            old_diffs_inv.add(row[i] - old_val)
            
        # New diffs involving this mark
        new_diffs_inv = set()
        for i in range(m_idx):
            new_diffs_inv.add(new_val - row[i])
        for i in range(m_idx + 1, len(row)):
            new_diffs_inv.add(row[i] - new_val)
            
        # Diffs in the row NOT involving this mark
        other_diffs_in_row = set()
        for i in range(len(row)):
            if i == m_idx: continue
            for j in range(i + 1, len(row)):
                if j == m_idx: continue
                other_diffs_in_row.add(row[j] - row[i])
                
        # Conflict check:
        # 1. New diffs must not be in other_diffs_in_row
        if new_diffs_inv & other_diffs_in_row:
            continue
            
        # 2. New diffs must not be in used_diffs (excluding old_diffs_inv)
        # global_conflicts = new_diffs_inv & (used_diffs - old_diffs_inv)
        # Optimized:
        conflicts = new_diffs_inv & used_diffs
        if conflicts - old_diffs_inv:
            continue
            
        # Valid move!
        
        # Calculate energy change
        # The global max only changes if m_idx == k (the last mark)
        if m_idx == k:
            old_row_max = old_val
            new_row_max = new_val
            
            if new_row_max < old_row_max:
                # Recalculate global max
                new_global_max = 0
                for i, r in enumerate(rows):
                    if i == r_idx:
                        r_max = new_row_max
                    else:
                        r_max = r[k] # Since row is sorted, max is last element
                    if r_max > new_global_max:
                        new_global_max = r_max
                delta_E = new_global_max - current_max
            elif new_row_max > old_row_max:
                if new_row_max > current_max:
                    delta_E = new_row_max - current_max
                else:
                    delta_E = 0
            else:
                delta_E = 0
        else:
            # Internal move, global max unchanged
            delta_E = 0
            
        # Acceptance
        accepted = False
        if delta_E < 0:
            accepted = True
        else:
            if delta_E == 0:
                accepted = True # Accept neutral moves to explore
            else:
                if random.random() < math.exp(-delta_E / T):
                    accepted = True
                    
        if accepted:
            # Update state
            row[m_idx] = new_val
            # Update used_diffs
            used_diffs -= old_diffs_inv
            used_diffs |= new_diffs_inv
            
            # Update current_max
            if m_idx == k:
                if new_val < old_val:
                    # Recalculate global max
                    current_max = 0
                    for i, r in enumerate(rows):
                        if i == r_idx:
                            r_max = new_val
                        else:
                            r_max = r[k]
                        if r_max > current_max:
                            current_max = r_max
                elif new_val > old_val:
                    if new_val > current_max:
                        current_max = new_val
                        
            # Update best
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                
        T *= cooling
        
    return best_rows
