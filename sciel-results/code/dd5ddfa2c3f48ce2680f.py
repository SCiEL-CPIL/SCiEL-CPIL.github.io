import random
import time
import math

def construct(n_rows, k):
    # 1. Generate initial solution using Greedy
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
        
    # 2. Simulated Annealing
    # We maintain a dictionary of difference counts to calculate conflicts efficiently.
    # conflict_count = sum(max(0, count - 1) for count in diff_counts.values())
    
    def build_diff_counts(rgs):
        dc = {}
        for r in rgs:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    dc[d] = dc.get(d, 0) + 1
        return dc

    def get_conflict_count(dc):
        return sum(max(0, c - 1) for c in dc.values())

    current_rows = [r[:] for r in rows]
    current_dc = build_diff_counts(current_rows)
    current_conflicts = get_conflict_count(current_dc)
    current_max = max(max(r) for r in current_rows)
    current_energy = current_max + 1000 * current_conflicts # Penalty for conflicts
    
    best_rows = [r[:] for r in current_rows]
    best_dc = {k: v for k, v in current_dc.items()}
    best_conflicts = current_conflicts
    best_max = current_max
    best_energy = current_energy
    
    # SA Parameters
    T = 100.0
    cooling = 0.999
    min_T = 1e-4
    start_time = time.time()
    time_limit = 110 # Leave some buffer for 120s limit
    
    # Pre-compute indices for faster access?
    # rows[r][j] neighbors are r[j-1] and r[j+1]
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k) # 0 is fixed at 0
        
        row = current_rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new_val
        low = row[j_idx - 1] + 1
        high = row[j_idx + 1] - 1 if j_idx < k else old_val + 10 # If last element, we can try to increase? 
        # Actually, we want to minimize max, so increasing the last element is usually bad unless it allows reducing others.
        # But for simplicity, let's restrict to [low, high] where high is constrained by the next element.
        # If j_idx == k, there is no next element in the row. We can try any value > low.
        # But to keep the search bounded and focused on minimization, let's restrict the range.
        # If j_idx < k:
        #   high = row[j_idx + 1] - 1
        # else:
        #   high = old_val + 10 # Allow small increases, or just old_val? 
        # Actually, if we allow increases, we might find a configuration where a later element can be decreased significantly.
        # Let's allow a small range of increase for the last element, or any element if it helps reduce conflicts.
        # However, the energy function penalizes max. So increasing max is bad.
        # Let's stick to:
        if j_idx < k:
            high = row[j_idx + 1] - 1
        else:
            # For the last element, we can try to decrease it or keep it.
            # Increasing it only makes sense if it allows reducing a previous element? No, previous elements are smaller.
            # So for the last element, we should only try to decrease it or keep it.
            # But wait, if we decrease it, it must be > row[j_idx-1].
            # So high = old_val.
            high = old_val
            
        if low > high:
            T *= cooling
            continue
            
        # Pick new_val
        # To make it faster, we can pick randomly from [low, high]
        new_val = random.randint(low, high)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        # Calculate change in differences
        # Remove old diffs
        old_diffs = []
        for i in range(k + 1):
            if i == j_idx:
                continue
            d = abs(old_val - row[i])
            old_diffs.append(d)
            
        # Add new diffs
        new_diffs = []
        for i in range(k + 1):
            if i == j_idx:
                continue
            d = abs(new_val - row[i])
            new_diffs.append(d)
        
        # Update diff counts and conflicts
        # We need to track the change in conflict count.
        # conflict_count = sum(max(0, count - 1))
        # Let's calculate delta_conflicts
        
        # Helper to get delta conflict for a set of diffs being removed/added
        # This is a bit complex to do incrementally without updating the dict.
        # Given the small size (k+1 <= 9, n_rows <= 4), we can just update the dict and recalculate?
        # No, recalculating is O(N^2) which is fine.
        
        # Let's update current_dc temporarily
        # We need to be careful to revert if rejected.
        
        # Save state
        saved_dc = {k: v for k, v in current_dc.items()}
        saved_row_val = old_val
        
        # Apply change
        for d in old_diffs:
            current_dc[d] -= 1
            if current_dc[d] == 0:
                del current_dc[d]
                
        for d in new_diffs:
            current_dc[d] = current_dc.get(d, 0) + 1
            
        row[j_idx] = new_val
        
        new_conflicts = get_conflict_count(current_dc)
        new_max = max(max(r) for r in current_rows)
        new_energy = new_max + 1000 * new_conflicts
        
        delta_E = new_energy - current_energy
        
        # Acceptance
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_energy = new_energy
            current_conflicts = new_conflicts
            current_max = new_max
            
            # Update best if valid and better
            if new_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
                best_dc = {k: v for k, v in current_dc.items()}
                best_conflicts = new_conflicts
        else:
            # Revert
            row[j_idx] = saved_row_val
            current_dc = saved_dc
            # current_energy, current_conflicts, current_max remain unchanged
            
        T *= cooling
        
    # Return the best valid solution
    # Ensure best_rows is valid
    return best_rows
