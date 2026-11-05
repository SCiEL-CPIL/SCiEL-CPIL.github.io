import time
import random
import math
from typing import List, Set, Tuple

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set of n_rows rows, each with k+1 elements.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Greedy Initialization with Randomization
    def greedy_construct(n_rows, k, noise=0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Try to build the row
            # We want to keep values small.
            # Standard greedy: pick smallest x such that {x-a for a in row} are not in used.
            # To diversify, we can sometimes skip the smallest valid x.
            x = 0
            while len(row) < k + 1:
                x += 1
                # Add some noise: with small prob, skip a few values
                if noise > 0 and random.random() < 0.1:
                    x += random.randint(1, noise)
                
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # 2. Local Search (Simulated Annealing)
    def calculate_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows, used_diffs):
        # Check if all differences in rows are distinct and match used_diffs
        # This is used for verification, but in SA we update incrementally.
        pass

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # Generate initial best solution
    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy starts
    for i in range(5):
        rows = greedy_construct(n_rows, k, noise=i)
        scope = calculate_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
        
        if time.time() > deadline - 10:
            break

    if best_rows is None:
        # Fallback: simple greedy
        best_rows = greedy_construct(n_rows, k, noise=0)
        best_scope = calculate_scope(best_rows)

    # Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Pre-compute initial used diffs
    current_used = set()
    for r in current_rows:
        current_used |= get_diffs(r)
        
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        if T < T_min:
            break
            
        # Pick a random row and a random position to modify
        r_idx = random.randint(0, n_rows - 1)
        # Positions 1..k (0 is fixed at 0)
        pos_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][pos_idx]
        
        # Propose a new value
        # Bias towards smaller values to reduce scope
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure strictly increasing order
        # If pos_idx > 1, new_val must be > current_rows[r_idx][pos_idx-1]
        # If pos_idx < k, new_val must be < current_rows[r_idx][pos_idx+1]
        if pos_idx > 1 and new_val <= current_rows[r_idx][pos_idx-1]:
            continue
        if pos_idx < k and new_val >= current_rows[r_idx][pos_idx+1]:
            continue
        if new_val <= 0: # Must be positive if it's not the first element? 
            # Actually, elements are strictly increasing starting at 0, so next is >= 1.
            continue

        # Create new row
        new_row = current_rows[r_idx][:]
        new_row[pos_idx] = new_val
        
        # Check validity
        old_diffs = get_diffs(current_rows[r_idx])
        new_diffs = get_diffs(new_row)
        
        # Remove old diffs, check for conflicts with new diffs
        # The global set is current_used.
        # We remove old_diffs from current_used, then check if new_diffs intersect with the remaining current_used.
        # Also check if new_diffs have internal duplicates (they won't, as row is strictly increasing).
        
        # Efficient check:
        # 1. Remove old_diffs from a temporary set? No, just check intersection.
        # Conflicts occur if:
        # a) Any d in new_diffs is in (current_used - old_diffs)
        # b) Any d in new_diffs is in old_diffs (but d != old_val - a for the specific pair? No, if d is in old_diffs and new_diffs, it's fine ONLY if the pair generating it is the same? 
        #    Wait. The set of differences for the row changes.
        #    Let D_old be the set of diffs for the old row.
        #    Let D_new be the set of diffs for the new row.
        #    The global set of used diffs is U.
        #    We require that (U \ D_old) \cap D_new is empty.
        #    Also, D_new must not contain duplicates (guaranteed by strictly increasing).
        
        # So:
        # U_without_old = U - D_old
        # If U_without_old & D_new is not empty -> Invalid.
        
        # Optimization: We don't want to create a new set U_without_old every time.
        # We can check: for d in D_new:
        #   if d in U and d not in D_old: return False
        
        valid = True
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                valid = False
                break
        
        if not valid:
            # Reheat occasionally
            if random.random() < 0.01:
                T = max(T, 10.0)
            continue
            
        # Calculate new scope
        new_scope = max(max(current_rows[i]) if i != r_idx else max(new_row) for i in range(n_rows))
        
        delta_scope = new_scope - current_scope
        
        # Acceptance probability
        if delta_scope <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_scope / T)
            
        if accept:
            # Update state
            current_used -= old_diffs
            current_used |= new_diffs
            current_rows[r_idx] = new_row
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                
        # Cool down
        T *= cooling
        
        # Periodic reheat to escape local minima
        if step % 1000 == 0 and T < 5.0:
            T = 10.0

    return best_rows

# Wrapper to ensure the function signature matches exactly
def construct(n_rows, k):
    return _construct_impl(n_rows, k)

def _construct_impl(n_rows, k):
    # The actual implementation above
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def greedy_construct(n_rows, k, noise=0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if noise > 0 and random.random() < 0.1:
                    x += random.randint(1, noise)
                
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def calculate_scope(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy starts
    for i in range(10):
        if time.time() > deadline - 20:
            break
        rows = greedy_construct(n_rows, k, noise=i)
        scope = calculate_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        best_rows = greedy_construct(n_rows, k, noise=0)
        best_scope = calculate_scope(best_rows)

    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    current_used = set()
    for r in current_rows:
        current_used |= get_diffs(r)
        
    T = 100.0  # Higher initial temp for better exploration
    T_min = 0.1
    cooling = 0.9995
    
    step = 0
    last_best_time = time.time()
    
    while time.time() < deadline - 5:
        step += 1
        if T < T_min:
            # If stuck, try to reheat
            if time.time() - last_best_time > 5:
                 T = 20.0
            else:
                 break
        
        r_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][pos_idx]
        
        # Proposal: change value
        # Bias towards reducing scope if current value is the max
        if old_val == current_scope:
             delta = random.randint(-10, 2)
        else:
             delta = random.randint(-5, 5)
             
        new_val = old_val + delta
        
        if pos_idx > 1 and new_val <= current_rows[r_idx][pos_idx-1]:
            continue
        if pos_idx < k and new_val >= current_rows[r_idx][pos_idx+1]:
            continue
        if new_val <= 0:
            continue

        new_row = current_rows[r_idx][:]
        new_row[pos_idx] = new_val
        
        old_diffs = get_diffs(current_rows[r_idx])
        new_diffs = get_diffs(new_row)
        
        valid = True
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                valid = False
                break
        
        if not valid:
            continue
            
        new_scope = max(max(current_rows[i]) if i != r_idx else max(new_row) for i in range(n_rows))
        
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_scope / T)
            
        if accept:
            current_used -= old_diffs
            current_used |= new_diffs
            current_rows[r_idx] = new_row
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                last_best_time = time.time()
                
        T *= cooling
        
        # Hard reheat every 5000 steps if no improvement
        if step % 5000 == 0 and T < 5.0:
            T = 15.0

    return best_rows
