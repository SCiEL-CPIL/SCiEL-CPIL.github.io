import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s (or 90s kill, but 60s is stated as limit for evaluation)
    # Prompt says "Each evaluation must finish within 60 s".
    # "evaluator KILLS the program after 90s".
    # So we have 60s to run.
    deadline = start_time + 55.0
    
    # Initial Set: Seed
    # A = {64, ..., 128}
    A = set(range(64, 129))
    A_list = list(A)
    
    # Counts array for sums 0..256
    counts = np.zeros(257, dtype=np.int32)
    
    # Function to compute initial counts
    for a in A_list:
        for b in A_list:
            if a <= b:
                counts[a + b] += 1
                
    # Helper to get current metrics
    def get_metrics(counts):
        target = counts[128:257]
        peak = target.max()
        zeros = (target == 0).sum()
        sum_sq = (target ** 2).sum()
        return peak, zeros, sum_sq

    curr_peak, curr_zeros, curr_sum_sq = get_metrics(counts)
    best_peak = curr_peak
    best_sum_sq = curr_sum_sq
    best_A = set(A)
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.9995
    
    # Precompute all possible y values
    all_vals = list(range(129))
    
    steps = 0
    while time.time() < deadline:
        # 1. Select x to remove
        idx_x = random.randint(0, len(A_list) - 1)
        x = A_list[idx_x]
        
        # 2. Select y to add
        # Pick random y not in A
        y = random.choice(all_vals)
        while y in A:
            y = random.choice(all_vals)
        
        # 3. Apply Move: Remove x, Add y
        # Track changes to revert
        changes = []
        
        # Remove x
        for a in A_list:
            s = x + a
            old_c = counts[s]
            counts[s] = old_c - 1
            changes.append((s, old_c))
            
        # Add y
        for a in A_list:
            s = y + a
            old_c = counts[s]
            counts[s] = old_c + 1
            changes.append((s, old_c))
            
        s = 2 * y
        old_c = counts[s]
        counts[s] = old_c + 1
        changes.append((s, old_c))
        
        # Update A and A_list
        A.remove(x)
        # Swap remove for A_list
        A_list[idx_x] = A_list[-1]
        A_list.pop()
        A.add(y)
        A_list.append(y)
        
        # 4. Evaluate New State
        new_peak, new_zeros, new_sum_sq = get_metrics(counts)
        
        # 5. Acceptance Criteria
        # Primary: Minimize Peak
        # Secondary: Minimize Sum of Squares (flatten)
        # Constraint: Valid (zeros == 0)
        
        if new_zeros > 0:
            new_cost = 1000 * new_zeros + new_peak + 0.1 * new_sum_sq
        else:
            new_cost = new_peak + 0.1 * new_sum_sq
            
        if curr_zeros > 0:
            curr_cost = 1000 * curr_zeros + curr_peak + 0.1 * curr_sum_sq
        else:
            curr_cost = curr_peak + 0.1 * curr_sum_sq
            
        delta_cost = new_cost - curr_cost
        
        accepted = False
        if delta_cost < 0:
            accepted = True
        else:
            # Boltzmann probability
            if T > 1e-6:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            curr_peak = new_peak
            curr_zeros = new_zeros
            curr_sum_sq = new_sum_sq
            if new_zeros == 0 and (new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq)):
                best_peak = new_peak
                best_sum_sq = new_sum_sq
                best_A = set(A)
        else:
            # Revert
            for s, old_c in changes:
                counts[s] = old_c
            
            # Revert A and A_list
            A.remove(y)
            A_list.pop()
            A_list.append(x)
            # Note: x was swapped to last position during removal.
            # We need to restore x to its original position?
            # Actually, order doesn't matter for the set logic, only for the list consistency.
            # But we swapped x to the end and popped.
            # So A_list is now A \ {x} + [y].
            # To revert: Remove y (last), add x.
            # This restores the set content.
            # The order in A_list is changed, but that's fine.
            A.add(x)
            # Wait, if I pop y, A_list is A \ {x}.
            # Then I append x.
            # A_list is A \ {x} + [x].
            # This is correct.
            
        # Cooling
        T *= cooling
        steps += 1
        
    return list(best_A)
