"""Difference triangle set (4,7) - Simulated Annealing approach."""
import random
import time

def _get_diffs(row):
    """Compute all pairwise differences for a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _check_validity(rows, k):
    """
    Check if the set of rows is valid.
    Returns (is_valid, max_entry).
    """
    all_diffs = set()
    max_entry = 0
    expected_count = len(rows) * k * (k + 1) // 2
    
    for row in rows:
        if len(row) != k + 1:
            return False, float('inf')
        if row[0] != 0:
            return False, float('inf')
        if not all(row[i] < row[i+1] for i in range(len(row)-1)):
            return False, float('inf')
            
        row_max = max(row)
        if row_max > max_entry:
            max_entry = row_max
            
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False, float('inf')
                all_diffs.add(d)
                
    if len(all_diffs) != expected_count:
        return False, float('inf')
        
    return True, max_entry

def _init_solution(n_rows, k, max_val=200):
    """
    Initialize a valid solution using randomized greedy.
    """
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = [0]
        # Randomize the order of trying candidates to get diverse starts
        candidates = list(range(1, max_val))
        random.shuffle(candidates)
        
        # Greedy fill
        for _ in range(k):
            found = False
            for x in candidates:
                # Check if x can be added
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                return None # Failed to construct
            
        rows.append(row)
        
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_rows = None
    best_score = float('inf')
    
    # Multi-start Simulated Annealing
    num_restarts = 5
    
    for restart in range(num_restarts):
        # 1. Initialize
        current_rows = _init_solution(n_rows, k, max_val=150)
        if current_rows is None:
            continue
            
        current_valid, current_score = _check_validity(current_rows, k)
        if not current_valid:
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_rows = [r[:] for r in current_rows]
            
        # 2. Simulated Annealing
        temperature = 10.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        while temperature > min_temp and time.time() - start_time < time_limit:
            # Pick a random row and a random element to perturb
            r_idx = random.randint(0, n_rows - 1)
            # Do not perturb the first element (must be 0)
            e_idx = random.randint(1, k)
            
            # Generate a perturbation
            # Try shifting the element by a small amount
            delta = random.randint(-3, 3)
            if delta == 0:
                continue
                
            old_val = current_rows[r_idx][e_idx]
            new_val = old_val + delta
            
            # Basic bounds check
            if new_val <= 0:
                continue
            if e_idx > 0 and new_val <= current_rows[r_idx][e_idx - 1]:
                continue
            if e_idx < k and new_val >= current_rows[r_idx][e_idx + 1]:
                continue
            
            # Apply perturbation
            current_rows[r_idx][e_idx] = new_val
            
            # Check validity
            new_valid, new_score = _check_validity(current_rows, k)
            
            if new_valid:
                if new_score < current_score:
                    # Accept improvement
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Accept worse with probability exp(-delta_E / T)
                    # Here objective is minimization, so delta_E = new_score - current_score > 0
                    delta_e = new_score - current_score
                    if random.random() < pow(2.71828, -delta_e / temperature):
                        current_score = new_score
                    else:
                        # Revert
                        current_rows[r_idx][e_idx] = old_val
            else:
                # Invalid move: revert
                current_rows[r_idx][e_idx] = old_val
                
            temperature *= cooling_rate
            
    return best_rows
