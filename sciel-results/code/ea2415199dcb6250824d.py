import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Lower bound for scope
    lower_bound = total_diffs_needed
    
    # Helper to get all differences in a row
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper to check validity and count conflicts
    def check_state(rows):
        """
        Returns:
        - valid: bool
        - max_scope: int
        - conflict_count: int (number of duplicate differences)
        """
        all_diffs = []
        max_scope = 0
        for row in rows:
            if not row:
                continue
            max_scope = max(max_scope, max(row))
            all_diffs.extend(get_row_diffs(row))
        
        if len(all_diffs) != len(set(all_diffs)):
            # Count duplicates
            from collections import Counter
            counts = Counter(all_diffs)
            conflict_count = sum(count - 1 for count in counts.values() if count > 1)
            return False, max_scope, conflict_count
        return True, max_scope, 0

    # Generate a random valid row with max element <= M
    # This is a simple greedy with backtracking for a single row
    def generate_random_row(M, used_diffs, rng):
        row = [0]
        # We need to pick k more numbers
        # To ensure we don't get stuck, we pick from available candidates
        
        # Sort candidates by some heuristic? 
        # Let's just pick random valid candidates from [last+1, M]
        
        # To make it faster, we can pre-filter or just try
        # Since M is small initially or we start large, this is okay.
        
        # Actually, for SA, we want to start with a valid configuration.
        # We can construct rows sequentially with a large M first.
        
        for _ in range(k):
            last_val = row[-1]
            candidates = list(range(last_val + 1, M + 1))
            rng.shuffle(candidates)
            
            found = False
            for x in candidates:
                new_diffs = [x - y for y in row]
                if not any(d in used_diffs for d in new_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                return None
        return row

    # Initialize with a large scope to ensure validity
    # Start with M = lower_bound + 50 to be safe
    current_M = lower_bound + 50
    rng = random.Random(42)
    
    rows = []
    used_diffs = set()
    success = False
    
    # Try to construct a valid initial state
    while not success:
        rows = []
        used_diffs = set()
        for i in range(n_rows):
            r = generate_random_row(current_M, used_diffs, rng)
            if r is None:
                break
            rows.append(r)
        if len(rows) == n_rows:
            success = True
        else:
            # If failed, increase M
            current_M += 10
            if current_M > 500:
                # Fallback: just return something valid even if large
                # This shouldn't happen for (7,5)
                break
    
    if not success:
        # Emergency fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Simulated Annealing
    # State: rows
    # Energy: conflict_count * 1000 + max_scope
    # We want to minimize energy.
    # Moves:
    # 1. Pick a random row, pick a random element, move it to a random new position (maintaining order).
    # 2. Pick a random row, pick a random element, shift it by +/- 1.
    
    current_rows = [r[:] for r in rows]
    current_valid, current_scope, current_conflicts = check_state(current_rows)
    current_energy = current_conflicts * 1000 + current_scope
    
    best_rows = [r[:] for r in rows]
    best_energy = current_energy
    best_scope = current_scope
    best_valid = current_valid
    
    T = 100.0
    cooling_rate = 0.999
    
    # We will iterate until time runs out
    while time.time() < DEADLINE:
        # Pick a move
        move_type = rng.random()
        row_idx = rng.randrange(n_rows)
        row = current_rows[row_idx]
        
        new_rows = [r[:] for r in current_rows]
        
        if move_type < 0.5:
            # Shift an element
            if len(row) <= 1:
                continue
            # Pick an index to shift (not 0, as 0 is fixed)
            idx = rng.randrange(1, len(row))
            delta = rng.choice([-1, 1])
            new_val = row[idx] + delta
            
            # Check bounds
            if new_val <= row[idx-1] or (idx == len(row)-1 and new_val > 1000):
                continue
            if new_val < 0:
                continue
            
            # Check if new_val causes conflicts within the same row
            # We need to check all pairs involving idx
            # For simplicity, just update and check global validity? 
            # That's expensive. 
            # Let's do a quick local check first.
            # Actually, for SA, we can just compute the full energy difference if we are smart,
            # but for simplicity and robustness, we'll just generate the new state and check.
            # To speed up, we can limit the search space.
            
            new_rows[row_idx][idx] = new_val
            
            # Ensure strictly increasing
            if not all(new_rows[row_idx][i] < new_rows[row_idx][i+1] for i in range(len(new_rows[row_idx])-1)):
                continue
                
        elif move_type < 0.8:
            # Swap two elements? No, must maintain order.
            # Instead, pick two indices and try to swap their values? No.
            # Let's do a "jump": pick an element and set it to a random value between neighbors.
            if len(row) <= 1:
                continue
            idx = rng.randrange(1, len(row))
            lower = row[idx-1] + 1
            upper = row[idx+1] - 1 if idx < len(row)-1 else 1000
            if lower > upper:
                continue
            new_val = rng.randint(lower, upper)
            new_rows[row_idx][idx] = new_val
            
        else:
            # Cross-row swap of a difference? Complex.
            # Let's just do another shift.
            if len(row) <= 1:
                continue
            idx = rng.randrange(1, len(row))
            # Random jump
            new_val = row[idx] + rng.randint(-5, 5)
            if new_val <= row[idx-1] or (idx < len(row)-1 and new_val >= row[idx+1]):
                continue
            new_rows[row_idx][idx] = new_val

        # Evaluate new state
        new_valid, new_scope, new_conflicts = check_state(new_rows)
        new_energy = new_conflicts * 1000 + new_scope
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            # Accept
            current_rows = new_rows
            current_valid = new_valid
            current_scope = new_scope
            current_conflicts = new_conflicts
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_rows = [r[:] for r in rows] # Wait, copy current
                best_rows = [r[:] for r in current_rows]
                best_scope = new_scope
                best_valid = new_valid
        else:
            # Accept with probability
            if T > 0:
                prob = pow(2.71828, -delta_e / T)
                if rng.random() < prob:
                    current_rows = new_rows
                    current_valid = new_valid
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    current_energy = new_energy
        
        T *= cooling_rate
        
        # If we have a valid solution, we can try to reduce scope further
        # The energy function already penalizes scope, so it should naturally do this.
        
        # Optimization: If we are valid and scope is low, we can tighten the search.
        # But for now, let SA run.

    # Return the best valid solution found
    if best_valid:
        return best_rows
    else:
        # If no valid solution was found during SA (unlikely if init was valid),
        # return the initial valid one or the best attempt
        if current_valid:
            return current_rows
        else:
            # Return initial rows if they were valid
            if check_state(rows)[0]:
                return rows
            # Otherwise, fallback
            rows_fb = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows_fb.append(row)
            return rows_fb

# EVOLVE-BLOCK-END
