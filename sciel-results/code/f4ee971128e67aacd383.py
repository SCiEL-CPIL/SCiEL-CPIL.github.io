"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed):
        """Multi-strategy greedy with randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                found = False
                # Try a range of candidate positions with some randomness
                jump = rng.randint(1, 40)
                x = last + jump
                for candidate in range(max(last + 1, x - 15), x + 20):
                    nd = [candidate - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(candidate)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    # Fallback: linear search
                    for candidate in range(last + 1, last + 300):
                        nd = [candidate - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(candidate)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def local_compact(rows, time_limit):
        """Iteratively compact each element to its smallest feasible value."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                # Process from right to left for better compaction
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    upper = row[pos + 1] if pos < L - 1 else old_val
                    # Try to move this element left
                    for new_val in range(lower, old_val):
                        if new_val >= upper:
                            break
                        # Check if this placement is valid
                        valid = True
                        # Remove old differences involving this position
                        old_diffs = set()
                        for i in range(L):
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            elif i > pos:
                                old_diffs.add(row[i] - old_val)
                        
                        # Check new differences
                        new_diffs = set()
                        for i in range(L):
                            if i < pos:
                                d = new_val - row[i]
                                new_diffs.add(d)
                            elif i > pos:
                                d = row[i] - new_val
                                new_diffs.add(d)
                        
                        # All new diffs must not be in used_set minus old_diffs
                        other = used_set - old_diffs
                        if new_diffs & other:
                            valid = False
                            break
                        
                        if valid:
                            # Apply the change
                            used_set -= old_diffs
                            used_set |= new_diffs
                            row[pos] = new_val
                            improved = True
                            break
        return work

    def simulated_annealing(rows_init, time_limit):
        """Simulated annealing to escape local optima."""
        import math
        rng = random.Random(int(time.time() * 1000) % 10000)
        current = [list(r) for r in rows_init]
        current_scope = get_scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        
        temperature = max(10.0, current_scope * 0.1)
        cooling_rate = 0.995
        min_temperature = 0.1
        
        while time.time() < time_limit and temperature > min_temperature:
            # Pick a random row and position to perturb
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = current[r_idx]
            
            # Generate a new value
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < L - 1 else row[pos] + 50
            new_val = rng.randint(lower, min(upper, row[pos] + 100))
            
            if new_val <= row[pos - 1] or (pos < L - 1 and new_val >= row[pos + 1]):
                temperature *= cooling_rate
                continue
            
            old_val = row[pos]
            
            # Calculate the change in scope
            old_max = max(row)
            new_row = row.copy()
            new_row[pos] = new_val
            new_max = max(new_row)
            old_scope = get_scope(current)
            new_scope = max(max(r) for i, r in enumerate(current) if i != r_idx) 
            new_scope = max(new_scope, new_max)
            
            delta = new_scope - old_scope
            
            # Check validity
            used_set = build_used(current)
            # Remove old diffs
            for i in range(L):
                if i < pos:
                    used_set.discard(old_val - row[i])
                elif i > pos:
                    used_set.discard(row[i] - old_val)
            
            # Add new diffs and check for conflicts
            valid = True
            for i in range(L):
                if i < pos:
                    d = new_val - row[i]
                    if d in used_set:
                        valid = False
                        break
                    used_set.add(d)
                elif i > pos:
                    d = row[i] - new_val
                    if d in used_set:
                        valid = False
                        break
                    used_set.add(d)
            
            if not valid:
                temperature *= cooling_rate
                continue
            
            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                row[pos] = new_val
                if new_scope < best_scope:
                    best = [list(r) for r in current]
                    best_scope = new_scope
            
            temperature *= cooling_rate
        
        return best, best_scope

    def backtrack_refine(rows_init, max_scope, time_limit):
        """MRV backtracking to try to find a solution with scope <= max_scope."""
        rows = [list(r) for r in rows_init]
        used = build_used(rows)
        
        def can_place(row, pos, new_val, used_set):
            if pos > 0 and new_val <= row[pos - 1]:
                return False
            if pos < L - 1 and new_val >= row[pos + 1]:
                return False
            old = set()
            for i in range(L):
                if i < pos:
                    old.add(row[pos] - row[i])
                elif i > pos:
                    old.add(row[i] - row[pos])
            other = used_set - old
            for i in range(L):
                if i < pos:
                    d = new_val - row[i]
                    if d in other:
                        return False
                elif i > pos:
                    d = row[i] - new_val
                    if d in other:
                        return False
            return True
        
        def apply_set(row, pos, new_val, used_set):
            old = set()
            for i in range(L):
                if i < pos:
                    old.add(row[pos] - row[i])
                elif i > pos:
                    old.add(row[i] - row[pos])
            for d in old:
                used_set.discard(d)
            for i in range(L):
                if i < pos:
                    used_set.add(new_val - row[i])
                elif i > pos:
                    used_set.add(row[i] - new_val)
            row[pos] = new_val
        
        # Try to compact each row
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    upper = row[pos + 1] - 1 if pos < L - 1 else old_val - 1
                    if lower > upper:
                        continue
                    for new_val in range(lower, upper + 1):
                        if new_val >= old_val:
                            break
                        if can_place(row, pos, new_val, used):
                            apply_set(row, pos, new_val, used)
                            improved = True
                            break
        return rows

    # Main search loop
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        seed = num_restarts * 7919 + 13
        
        # Generate initial solution
        rows = greedy_construct(seed)
        if rows is None:
            continue
        
        # Validate
        if not is_valid(rows):
            continue
        
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        # Local compaction
        if time.time() < deadline - 2:
            compacted = local_compact(rows, deadline - 2)
            if is_valid(compacted):
                c_scope = get_scope(compacted)
                if c_scope < best_scope:
                    best_scope = c_scope
                    best_rows = [list(r) for r in compacted]
        
        # Simulated annealing for larger improvements
        if time.time() < deadline - 5 and best_scope > lb:
            sa_rows, sa_scope = simulated_annealing(compacted, deadline - 5)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [list(r) for r in sa_rows]
        
        # Backtracking refinement
        if time.time() < deadline - 3 and best_rows is not None:
            refined = backtrack_refine(best_rows, best_scope, deadline - 3)
            if is_valid(refined):
                r_scope = get_scope(refined)
                if r_scope < best_scope:
                    best_scope = r_scope
                    best_rows = [list(r) for r in refined]
        
        # If we've found a good solution, do more focused search
        if best_scope <= lb + 20 and time.time() < deadline - 10:
            # Try to improve with more aggressive compaction
            for _ in range(3):
                if time.time() >= deadline - 10:
                    break
                compacted = local_compact(best_rows, deadline - 10)
                if is_valid(compacted):
                    c_scope = get_scope(compacted)
                    if c_scope < best_scope:
                        best_scope = c_scope
                        best_rows = [list(r) for r in compacted]
        
        # Early exit if we hit the theoretical lower bound
        if best_scope == lb:
            break
    
    if best_rows is None:
        # Fallback: return a valid greedy solution
        rows = greedy_construct(1)
        if rows is None:
            # Last resort: simple greedy
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < L:
                    x = row[-1] + 1
                    while True:
                        nd = [x - a for a in row]
                        if not (set(nd) & used):
                            row.append(x)
                            used.update(nd)
                            break
                        x += 1
                rows.append(row)
            return rows
        return rows
    
    return best_rows
# EVOLVE-BLOCK-END
