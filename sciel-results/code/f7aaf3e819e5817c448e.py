import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 130.0  # 20s safety margin for 150s limit
    
    # Number of differences in one row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Initial greedy construction for upper bound
    best_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(best_rows)
    
    # Lower bound for M
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)
    
    # Try to find better solutions using simulated annealing on row representations
    # We'll iterate M from M_lower up to best_M - 1
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        rows = _solve_with_sa(n_rows, k, M, deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            break
    
    return best_rows

def _solve_with_sa(n_rows, k, M, deadline):
    """
    Solve using simulated annealing over row configurations.
    Each row is a sorted tuple of k+1 integers starting at 0.
    Objective: minimize max entry while ensuring all differences are distinct.
    """
    # Precompute valid rows (all subsets of size k from {1,...,M}, with 0 prepended)
    # and their difference sets for fast lookup
    valid_rows = []
    row_diffs_list = []
    
    for combo in _combinations_range(1, M + 1, k):
        row = [0] + list(combo)
        d_set = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    valid = False
                    break
                d_set.add(d)
        if valid:
            valid_rows.append(tuple(row))
            row_diffs_list.append(d_set)
    
    if len(valid_rows) < n_rows:
        return None
    
    # Simulated annealing: start with random selection of n_rows rows
    # State: list of n_rows row indices into valid_rows
    # Energy: number of difference conflicts + penalty for max entry
    
    best_state = None
    best_energy = float('inf')
    
    n_attempts = 200
    for attempt in range(n_attempts):
        if time.time() > deadline - 2:
            break
        
        # Random initialization
        state = random.sample(range(len(valid_rows)), n_rows)
        used_diffs = set()
        conflicts = 0
        for idx in state:
            for d in row_diffs_list[idx]:
                if d in used_diffs:
                    conflicts += 1
                used_diffs.add(d)
        
        max_entry = max(max(valid_rows[idx]) for idx in state)
        energy = conflicts * 1000 + max_entry
        
        cur_state = state[:]
        cur_energy = energy
        cur_used = used_diffs.copy()
        
        if energy < best_energy:
            best_energy = energy
            best_state = state[:]
        
        if best_energy == max_entry:  # No conflicts
            break
        
        # Annealing
        temp = 100.0
        cooling = 0.995
        step_count = 0
        max_steps = 50000
        
        while step_count < max_steps and time.time() < deadline - 2:
            if random.random() < 0.3:
                # Swap two rows
                i, j = random.sample(range(n_rows), 2)
                if cur_state[i] == cur_state[j]:
                    step_count += 1
                    continue
                old_i, old_j = cur_state[i], cur_state[j]
                cur_state[i], cur_state[j] = old_j, old_i
                
                # Recompute energy delta
                new_cur_state = cur_state
                new_used = set()
                new_conflicts = 0
                for idx in new_cur_state:
                    for d in row_diffs_list[idx]:
                        if d in new_used:
                            new_conflicts += 1
                        new_used.add(d)
                new_max = max(max(valid_rows[idx]) for idx in new_cur_state)
                new_energy = new_conflicts * 1000 + new_max
                
                delta = new_energy - cur_energy
                if delta < 0 or random.random() < _boltzmann(delta, temp):
                    cur_energy = new_energy
                    cur_used = new_used
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                        best_state = cur_state[:]
                        if best_energy == best_max_entry(best_state, valid_rows):
                            break
                else:
                    cur_state[i], cur_state[j] = old_i, old_j
            else:
                # Replace one row with a random different row
                i = random.randint(0, n_rows - 1)
                old_idx = cur_state[i]
                new_idx = random.randint(0, len(valid_rows) - 1)
                if new_idx == old_idx:
                    step_count += 1
                    continue
                cur_state[i] = new_idx
                
                new_used = set()
                new_conflicts = 0
                for idx in cur_state:
                    for d in row_diffs_list[idx]:
                        if d in new_used:
                            new_conflicts += 1
                        new_used.add(d)
                new_max = max(max(valid_rows[idx]) for idx in cur_state)
                new_energy = new_conflicts * 1000 + new_max
                
                delta = new_energy - cur_energy
                if delta < 0 or random.random() < _boltzmann(delta, temp):
                    cur_energy = new_energy
                    cur_used = new_used
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                        best_state = cur_state[:]
                        if best_energy == best_max_entry(best_state, valid_rows):
                            break
                else:
                    cur_state[i] = old_idx
            
            step_count += 1
            temp *= cooling
        
        if best_energy == best_max_entry(best_state, valid_rows):
            break
    
    if best_state is not None:
        best_max = best_max_entry(best_state, valid_rows)
        if best_energy == best_max:  # No conflicts
            return [list(valid_rows[idx]) for idx in best_state]
    
    return None

def best_max_entry(state, valid_rows):
    return max(max(valid_rows[idx]) for idx in state)

def _boltzmann(delta, temp):
    if temp < 1e-10:
        return 0.0
    import math
    if delta / temp > 50:
        return 0.0
    return math.exp(-delta / temp)

def _combinations_range(start, end, k):
    """Generate combinations of k elements from [start, end)."""
    if k == 0:
        yield ()
        return
    if start >= end:
        return
    if k == 1:
        for i in range(start, end):
            yield (i,)
        return
    # Recursive
    for i in range(start, end - k + 1):
        for rest in _combinations_range(i + 1, end, k - 1):
            yield (i,) + rest

def _greedy_construct(n_rows, k):
    """
    Greedy construction to get an initial upper bound.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
