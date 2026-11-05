import random
import math
import time

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
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
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Data structures for fast cost update
    # diff_count[d] = number of times difference d appears
    diff_count = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diff_count[d] = diff_count.get(d, 0) + 1
    
    current_max = max(max(r) for r in rows)
    
    def get_conflicts():
        # Number of differences that appear more than once
        # Sum of (count - 1) for all counts > 1
        return sum(c - 1 for c in diff_count.values() if c > 1)
    
    current_conflicts = get_conflicts()
    
    # Cost function: We want to minimize current_max, but conflicts must be 0.
    # Penalty for conflicts is high.
    # cost = current_max + current_conflicts * 10000
    current_cost = current_max + current_conflicts * 10000
    
    start_time = time.time()
    time_limit = 115
    
    T = 50.0
    min_T = 0.1
    cooling = 0.9995
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Pick a random row and a random move
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Move Type 1: Change a single element
        if random.random() < 0.7:
            j_idx = random.randint(1, k)
            # Choose a delta
            # Larger deltas are allowed
            max_delta = 20
            delta = random.randint(-max_delta, max_delta)
            if delta == 0:
                continue
            new_val = row[j_idx] + delta
            
            # Check bounds: row[j_idx-1] < new_val < row[j_idx+1]
            left = row[j_idx-1]
            right = row[j_idx+1] if j_idx < k else float('inf')
            
            if new_val <= left or new_val >= right:
                continue
            
            # Calculate Delta Cost
            old_val = row[j_idx]
            
            # Differences affected:
            # (old_val - row[i]) for i < j_idx
            # (row[i] - old_val) for i > j_idx
            # (new_val - row[i]) for i < j_idx
            # (row[i] - new_val) for i > j_idx
            
            delta_conflicts = 0
            # We need to update diff_count temporarily or just calculate the change in conflict count.
            # It's easier to just update diff_count, calculate new cost, and revert if needed.
            
            # Update diff_count for old differences (decrement)
            for i in range(j_idx):
                d = old_val - row[i]
                diff_count[d] -= 1
            for i in range(j_idx + 1, len(row)):
                d = row[i] - old_val
                diff_count[d] -= 1
                
            # Update diff_count for new differences (increment)
            for i in range(j_idx):
                d = new_val - row[i]
                diff_count[d] = diff_count.get(d, 0) + 1
            for i in range(j_idx + 1, len(row)):
                d = row[i] - new_val
                diff_count[d] = diff_count.get(d, 0) + 1
                
            # Update max
            new_max = max(max(r) for r in rows) # This is O(N*K), might be slow. 
            # Optimization: Only max of this row changes, but other rows might have the global max.
            # To be safe and fast enough for small N, we can just track global max.
            # However, recalculating global max every step is O(N*K). N=4, K=8, so 32 ops. This is fine.
            
            # Recalculate conflicts is O(M) where M is number of unique diffs.
            # M is bounded by N*K*(K+1)/2 ~ 4*36 = 144 diffs. So O(144) is fine.
            
            new_conflicts = get_conflicts()
            new_cost = new_max + new_conflicts * 10000
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                row[j_idx] = new_val
                current_cost = new_cost
                current_conflicts = new_conflicts
                current_max = new_max
                if new_conflicts == 0 and new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                # Increment old diffs back
                for i in range(j_idx):
                    d = old_val - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
                for i in range(j_idx + 1, len(row)):
                    d = row[i] - old_val
                    diff_count[d] = diff_count.get(d, 0) + 1
                    
                # Decrement new diffs back
                for i in range(j_idx):
                    d = new_val - row[i]
                    diff_count[d] -= 1
                for i in range(j_idx + 1, len(row)):
                    d = row[i] - new_val
                    diff_count[d] -= 1
                    
        else:
            # Move Type 2: Shift suffix
            if k == 0:
                T *= cooling
                continue
            j_idx = random.randint(1, k)
            delta = random.randint(-10, 10)
            if delta == 0:
                continue
            
            # Check if shift is valid for all elements in suffix
            # row[j_idx-1] < row[j_idx] + delta
            # and row[i] + delta < row[i+1] + delta (strictly increasing is preserved if delta is constant)
            # So we only need to check the boundary at j_idx-1 and j_idx
            if row[j_idx-1] >= row[j_idx] + delta:
                continue
            
            # Calculate Delta Cost
            old_suffix = row[j_idx:]
            new_suffix = [v + delta for v in old_suffix]
            
            # Differences affected are those where at least one index is >= j_idx
            # 1. i < j_idx, j >= j_idx: (row[j] - row[i]) -> (row[j]+delta - row[i])
            # 2. i >= j_idx, j > i: (row[j] - row[i]) -> (row[j]+delta - (row[i]+delta)) = row[j] - row[i] (Unchanged!)
            
            # So only cross-boundary differences change.
            # This is much faster.
            
            # Remove old cross diffs
            for i in range(j_idx):
                for j in range(j_idx, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] -= 1
            
            # Add new cross diffs
            for i in range(j_idx):
                for j in range(j_idx, len(row)):
                    d = (row[j] + delta) - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
            
            # Apply change to row
            for j in range(j_idx, len(row)):
                row[j] += delta
                
            new_max = max(max(r) for r in rows)
            new_conflicts = get_conflicts()
            new_cost = new_max + new_conflicts * 10000
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                current_conflicts = new_conflicts
                current_max = new_max
                if new_conflicts == 0 and new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                for j in range(j_idx, len(row)):
                    row[j] -= delta
                
                # Remove new cross diffs
                for i in range(j_idx):
                    for j in range(j_idx, len(row)):
                        d = (row[j] + delta) - row[i]
                        diff_count[d] -= 1
                        
                # Add old cross diffs
                for i in range(j_idx):
                    for j in range(j_idx, len(row)):
                        d = row[j] - row[i]
                        diff_count[d] = diff_count.get(d, 0) + 1
        
        T *= cooling

    return best_rows
