<<<<<<< SEARCH
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    # Get greedy baseline
    best = _greedy(n_rows, k)
    best_scope = max(max(row) for row in best)
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Try backtracking for M from lower_bound to best_scope - 1
    for M in range(lower_bound, min(best_scope, lower_bound + 10)):
        if time.time() > deadline - 30:
            break
        time_limit = min(deadline - 30, time.time() + 15)
        result = _backtrack_search(n_rows, k, M, time_limit)
        if result is not None:
            best = result
            best_scope = M
            break
    
    # Local search to try to reduce scope further
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n_rows):
            new_row = _improve_row(i, best, n_rows, k, deadline)
            if new_row is not None:
                if max(new_row) < max(best[i]):
                    best[i] = new_row
                    best_scope = max(max(row) for row in best)
                    improved = True
                    break
        if not improved:
            # Randomized restart
            new_best = _randomized_greedy(n_rows, k, deadline)
            if new_best is not None:
                new_scope = max(max(row) for row in new_best)
                if new_scope < best_scope:
                    best = new_best
                    best_scope = new_scope
    
    return best


def _greedy(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, deadline):
    used, rows = set(), []
    for _ in range(n_rows):
        if time.time() > deadline:
            return None
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            if x > 200:
                return None
            new = {x - a for a in row}
            if not new & used:
                if random.random() < 0.02:
                    continue
                row.append(x)
                used |= new
    return rows


def _backtrack_search(n_rows, k, M, time_limit):
    used_diffs = set()
    rows = []
    
    def build_row(pos, current_row):
        if time.time() > time_limit:
            return None
        if pos == k + 1:
            return current_row[:]
        if pos == 0:
            return build_row(1, [0])
        
        prev = current_row[-1]
        for val in range(prev + 1, M + 1):
            if time.time() > time_limit:
                return None
            
            ok = True
            new_diffs = []
            for elem in current_row:
                d = val - elem
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            remaining = k + 1 - pos - 1
            if M - val < remaining:
                continue
            
            current_row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            result = build_row(pos + 1, current_row)
            if result is not None:
                return result
            
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return None
    
    for i in range(n_rows):
        row = build_row(0, [])
        if row is None:
            return None
        rows.append(row)
    
    return rows


def _improve_row(idx, rows, n_rows, k, deadline):
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                used_diffs.add(row[l] - row[j])
    
    old_max = max(rows[idx])
    
    for target in range(old_max - 1, 0, -1):
        if time.time() > deadline:
            return None
        result = _build_row_limited(k, target, used_diffs, deadline)
        if result is not None:
            return result
    
    return None


def _build_row_limited(k, max_val, used_diffs, deadline):
    row = [0]
    
    def bt(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        if pos == 0:
            return bt(1)
        
        prev = row[-1]
        # Limit candidates for speed
        for val in range(prev + 1, min(prev + 15, max_val + 1)):
            if time.time() > deadline:
                return False
            
            ok = True
            new_diffs = []
            for elem in row:
                d = val - elem
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            remaining = k + 1 - pos - 1
            if max_val - val < remaining:
                continue
            
            row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            if bt(pos + 1):
                return True
            
            row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return False
    
    if bt(1):
        return row[:]
    return None
=======
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    
    # Get greedy baseline
    best = _greedy(n_rows, k)
    best_scope = max(max(row) for row in best)
    
    # Simulated annealing to improve
    best = _simulated_annealing(best, n_rows, k, deadline)
    
    return best


def _greedy(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _compute_energy(rows, n_rows, k):
    """Compute energy: number of duplicate differences + scope penalty"""
    diff_count = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_count[d] = diff_count.get(d, 0) + 1
    
    conflicts = sum(max(0, c - 1) for c in diff_count.values())
    scope = max(max(row) for row in rows)
    
    # Energy = conflicts + scope * weight
    # We want to minimize both, but conflicts are more important
    return conflicts * 1000 + scope


def _simulated_annealing(initial_rows, n_rows, k, deadline):
    """Simulated annealing to minimize energy"""
    rows = [row[:] for row in initial_rows]
    current_energy = _compute_energy(rows, n_rows, k)
    best_rows = [row[:] for row in rows]
    best_energy = current_energy
    
    total_elements = n_rows * (k + 1)
    max_scope = max(max(row) for row in rows)
    
    T = 100.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        
        # Choose a random element to modify (not the first element of each row)
        row_idx = random.randint(0, n_rows - 1)
        elem_idx = random.randint(1, k)  # Skip index 0
        
        old_val = rows[row_idx][elem_idx]
        
        # Generate new value: random within bounds
        # Must maintain strict increase
        min_val = rows[row_idx][elem_idx - 1] + 1
        max_val = rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else max_scope + 50
        
        if min_val >= max_val:
            min_val = max_val - 1
        
        # Biased toward current value
        if random.random() < 0.7:
            delta = random.randint(-5, 5)
            new_val = max(min_val, min(max_val, old_val + delta))
        else:
            new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
        
        # Apply change
        rows[row_idx][elem_idx] = new_val
        
        # Compute new energy
        new_energy = _compute_energy(rows, n_rows, k)
        
        # Accept or reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < (1.0 / (1.0 + new_energy)) * (-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_rows = [row[:] for row in rows]
        else:
            # Revert
            rows[row_idx][elem_idx] = old_val
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final local optimization pass
    best_rows = _local_optimize(best_rows, n_rows, k, deadline)
    
    return best_rows


def _local_optimize(rows, n_rows, k, deadline):
    """Local optimization: try to reduce each row's max while maintaining validity"""
    for i in range(n_rows):
        if time.time() > deadline - 5:
            break
        
        # Try to reduce the last element
        for attempt in range(50):
            row = rows[i]
            old_max = row[-1]
            if old_max <= 1:
                break
            
            # Try a smaller value for the last element
            new_last = old_max - random.randint(1, 3)
            if new_last <= row[-2]:
                continue
            
            # Check if this creates conflicts
            test_row = row[:-1] + [new_last]
            if _is_row_valid(test_row, rows, i, k):
                rows[i] = test_row
            else:
                break
        
        # Try to reduce middle elements
        for j in range(k - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            
            old_val = rows[i][j]
            if old_val <= rows[i][j - 1] + 1:
                continue
            
            new_val = old_val - random.randint(1, 2)
            if new_val <= rows[i][j - 1]:
                continue
            
            test_row = rows[i][:]
            test_row[j] = new_val
            if _is_row_valid(test_row, rows, i, k):
                rows[i] = test_row
    
    return rows


def _is_row_valid(row, all_rows, row_idx, k):
    """Check if row is valid given all other rows"""
    # Check internal differences
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            # Check against other rows
            for other_idx, other_row in enumerate(all_rows):
                if other_idx == row_idx:
                    continue
                for a in range(len(other_row)):
                    for b in range(a + 1, len(other_row)):
                        if other_row[b] - other_row[a] == d:
                            return False
    return True
>>>>>>> REPLACE
