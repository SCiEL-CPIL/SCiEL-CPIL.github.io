import time
import math
import random
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # Constants
    M_max = 2000  # Safe upper bound for differences in reasonable solutions
    
    # 1. Greedy Initialization
    # We need n_rows rows, each of length k+1, starting at 0, strictly increasing.
    # All differences between any two elements in the same row must be distinct across ALL rows.
    
    used_diffs = set()
    rows = []
    
    # Greedy construction: For each row, pick smallest possible integers
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    # State for Local Search
    current_rows = [r[:] for r in rows]
    
    # Precompute diff counts for fast conflict checking
    # diff_counts[d] = number of times difference d appears in all rows
    diff_counts = [0] * (M_max + 1)
    
    def update_diffs(row, sign):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= M_max:
                    diff_counts[d] += sign
    
    # Initialize counts
    for r in current_rows:
        update_diffs(r, 1)
    
    # Calculate initial num_dupes
    # num_dupes is the count of difference values d where diff_counts[d] > 1
    num_dupes = sum(1 for c in diff_counts if c > 1)
    
    # Helper to apply a move and return delta_dupes
    # We simulate the move to calculate delta before committing
    def calc_delta_dupes(ri, ei, new_val):
        row = current_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return 0
        
        delta = 0
        
        # Remove old diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 2:
                    delta -= 1 # 2->1 removes a duplicate value
                # if c > 2, it's still a duplicate, so no change in count of "dup values"
                # if c == 1, it becomes 0, no change in dup values
        
        # Add new diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 1:
                    delta += 1 # 1->2 creates a duplicate value
        
        return delta
    
    def apply_move(ri, ei, new_val):
        nonlocal num_dupes
        row = current_rows[ri]
        old_val = row[ei]
        
        # Update diff counts
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                old_c = diff_counts[d]
                diff_counts[d] -= 1
                if old_c == 2 and diff_counts[d] == 1:
                    num_dupes -= 1
        
        current_rows[ri][ei] = new_val
        
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                old_c = diff_counts[d]
                diff_counts[d] += 1
                if old_c == 1 and diff_counts[d] == 2:
                    num_dupes += 1

    def get_scope():
        return max(max(r) for r in current_rows)

    # Local Search Phase 1: Minimize num_dupes
    # Phase 2: Minimize scope (only if num_dupes == 0)
    
    def local_search(max_steps=5000):
        nonlocal num_dupes, best_scope, best_rows
        
        steps = 0
        T = 10.0
        alpha = 0.99
        
        while steps < max_steps and time.time() < deadline:
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            
            row = current_rows[ri]
            old_val = row[ei]
            
            lower = row[ei-1] + 1
            upper = row[ei+1] - 1 if ei < k else best_scope + 10
            # Constrain search space to keep scope reasonable
            # If we are in phase 2 (num_dupes==0), we should be strict about scope
            # If phase 1, we can be looser
            if num_dupes == 0:
                search_upper = min(upper, best_scope)
            else:
                search_upper = min(upper, best_scope + 20)
                
            if lower > search_upper:
                continue
                
            # Sample new value
            # Bias towards smaller values to help scope minimization
            # Uniform is okay, but let's try a slight bias
            new_val = random.randint(lower, search_upper)
            
            if new_val == old_val:
                continue
                
            delta_dupes = calc_delta_dupes(ri, ei, new_val)
            
            accepted = False
            if num_dupes > 0:
                # Phase 1: Reduce conflicts
                if delta_dupes < 0:
                    accepted = True
                elif delta_dupes == 0:
                    # Neutral, accept with prob to escape local min
                    if random.random() < 0.05:
                        accepted = True
            else:
                # Phase 2: Minimize scope
                # Only accept if it doesn't increase the max element
                # Since we don't track exact global max change easily, 
                # we accept if new_val < old_val (likely reduces local max)
                # or if new_val == old_val (neutral)
                # If new_val > old_val, it might increase scope, so reject unless T is high
                if new_val < old_val:
                    accepted = True
                elif new_val == old_val:
                    accepted = True
                else:
                    # Allow small increases to escape local minima in scope
                    # Penalty is the increase in value
                    delta_scope = new_val - old_val
                    if random.random() < math.exp(-delta_scope / T):
                        accepted = True
            
            if accepted:
                apply_move(ri, ei, new_val)
                
                # Update best if valid
                if num_dupes == 0:
                    cur_scope = get_scope()
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [r[:] for r in current_rows]
            
            T *= alpha
            if T < 0.1:
                T = 0.1
            
            steps += 1

    # Iterated Local Search Loop
    # 1. Run local search until stuck or time limit
    # 2. Perturb (Kick)
    # 3. Repeat
    
    # Initial local search
    local_search(max_steps=100000)
    
    # If we already found the theoretical minimum or close to it, we can stop early?
    # Counting bound is 112. Best known is 134.
    # If best_scope <= 112, we are optimal (or at the bound).
    
    while time.time() < deadline - 2:
        # Check if we are stuck in Phase 1 (num_dupes > 0) or Phase 2 (num_dupes == 0)
        
        # Perform a "Kick"
        # Randomly displace a few elements in a random row
        # This will likely create conflicts, but it changes the landscape
        
        # Choose a row to perturb
        ri = random.randint(0, n_rows - 1)
        # Choose 2-3 elements to displace
        num_kicks = random.randint(2, min(3, k))
        
        # Save state to revert if kick makes things much worse? 
        # No, ILS usually accepts the kick and restarts local search from there.
        # But we must ensure the row remains strictly increasing.
        
        row = current_rows[ri]
        
        # To kick, we pick random indices and assign new random values in valid range
        # We need to be careful to maintain order.
        # Easiest way: pick a random valid row configuration? No, too hard.
        # Instead, pick random indices in increasing order, and assign random values 
        # that fit between neighbors.
        
        indices = sorted(random.sample(range(1, k+1), num_kicks))
        
        # We will update these indices one by one.
        # To maintain order, we should process them carefully.
        # Actually, just picking random values in [lower, upper] for each index
        # and then sorting the whole row? No, that changes the structure too much.
        
        # Simple Kick: Pick one element, move it to a random valid position.
        # Do this for num_kicks elements.
        
        for _ in range(num_kicks):
            ei = random.randint(1, k)
            old_val = current_rows[ri][ei]
            
            lower = current_rows[ri][ei-1] + 1
            upper = current_rows[ri][ei+1] - 1 if ei < k else best_scope + 20
            
            if lower > upper:
                continue
                
            # Pick a value far from current to ensure a "kick"
            # Uniform random in [lower, upper]
            new_val = random.randint(lower, upper)
            
            # Apply move regardless of delta (this is the perturbation)
            # But we must update diff_counts and num_dupes correctly
            apply_move(ri, ei, new_val)
            
            # Note: apply_move updates num_dupes.
            # The kick might increase num_dupes significantly.
            
        # Run local search again to repair and optimize
        # Use a larger budget for repair if num_dupes is high
        # But keep it bounded.
        local_search(max_steps=50000)

    return best_rows
