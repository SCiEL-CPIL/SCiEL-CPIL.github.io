import random
import time
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Counting bound: each row has k(k+1)/2 differences, all must be distinct
    # So we need at least n_rows * k(k+1)/2 distinct differences in [1, M]
    # Thus M >= n_rows * k(k+1)/2
    bound = n_rows * (k * (k + 1) // 2)
    
    # For (7,5): bound = 7 * 15 = 105. Best known is 112.
    # We'll search for solutions with M from bound upwards, but start SA at a reasonable M.
    
    # Helper: compute all differences in a row
    def row_diffs(row):
        """Return list of all positive differences in the row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def get_conflicts_and_scope(rows):
        """Compute total conflicts and max scope."""
        diff_counts = defaultdict(int)
        max_val = 0
        for row in rows:
            for x in row:
                if x > max_val:
                    max_val = x
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] += 1
        
        conflicts = 0
        for c in diff_counts.values():
            if c > 1:
                conflicts += c - 1
        return conflicts, max_val
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    all_diffs.append(row[j] - row[i])
        return len(all_diffs) == len(set(all_diffs))
    
    # Greedy construction as a baseline
    def greedy_construct(n_rows, k, M_max):
        """Greedy: build rows one at a time, picking smallest possible values."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                # Try to find next smallest value that doesn't create conflicts
                found = False
                for v in range(row[-1] + 1, M_max + 1):
                    # Check all new differences
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = v - a
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        row.append(v)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    # Try greedy first with various M values
    best_rows = None
    best_scope = float('inf')
    
    # Try greedy from bound to bound + 30
    for M in range(bound, bound + 31):
        if time.time() > deadline:
            break
        rows = greedy_construct(n_rows, k, M)
        if rows is not None:
            _, scope = get_conflicts_and_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if scope == bound:
                    return best_rows  # Can't do better than counting bound
    
    # If greedy didn't find optimal, use simulated annealing
    # Start from greedy solution if available, else random
    
    def random_row(m_max):
        """Generate a random valid row starting at 0 with k+1 elements."""
        if k == 0:
            return [0]
        if m_max < k:
            return list(range(k + 1))
        try:
            others = random.sample(range(1, m_max + 1), k)
        except ValueError:
            others = list(range(1, k + 1))
        return [0] + sorted(others)
    
    # Simulated annealing
    def simulated_annealing(n_rows, k, m_max, time_limit):
        """SA to find valid configuration with max value <= m_max."""
        # Initialize
        if best_rows is not None and best_scope <= m_max:
            cur_rows = [r[:] for r in best_rows]
            cur_conflicts, cur_scope = get_conflicts_and_scope(cur_rows)
        else:
            cur_rows = [random_row(m_max) for _ in range(n_rows)]
            for i in range(n_rows):
                cur_rows[i] = [0] + sorted(cur_rows[i][1:])
            cur_conflicts, cur_scope = get_conflicts_and_scope(cur_rows)
        
        T = 100.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < time_limit:
            # Pick a random row and position to change
            r_idx = random.randint(0, n_rows - 1)
            if k > 0:
                j_idx = random.randint(1, k)
            else:
                continue
            
            old_val = cur_rows[r_idx][j_idx]
            # New value: perturb around old or random
            if random.random() < 0.7:
                # Local perturbation
                step = random.randint(-5, 5)
                new_val = max(0, min(m_max, old_val + step))
            else:
                # Random restart for this position
                new_val = random.randint(0, m_max)
            
            cur_rows[r_idx][j_idx] = new_val
            cur_rows[r_idx].sort()
            
            # Check for duplicates in row
            if len(set(cur_rows[r_idx])) < k + 1:
                cur_rows[r_idx][j_idx] = old_val
                cur_rows[r_idx].sort()
                T *= cooling
                if T < T_min:
                    T = T_min
                continue
            
            new_conflicts, new_scope = get_conflicts_and_scope(cur_rows)
            
            # Objective: minimize conflicts first, then scope
            delta_conf = new_conflicts - cur_conflicts
            delta_scope = new_scope - cur_scope
            delta_e = delta_conf + 0.01 * delta_scope
            
            if delta_e <= 0 or random.random() < pow(2.71828, -delta_e / T):
                cur_conflicts = new_conflicts
                cur_scope = new_scope
            else:
                # Revert
                cur_rows[r_idx][j_idx] = old_val
                cur_rows[r_idx].sort()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return cur_rows, cur_conflicts, cur_scope
    
    # Run SA with decreasing m_max
    # Start from best_scope - 1 (or bound) and go down to bound
    current_m_max = min(best_scope - 1, bound + 20) if best_rows else bound + 20
    current_m_max = max(current_m_max, bound)
    
    while time.time() < deadline and current_m_max >= bound:
        # Run SA for a few seconds
        sa_time = min(10.0, deadline - time.time())
        if sa_time <= 0:
            break
        
        rows, conflicts, scope = simulated_annealing(n_rows, k, current_m_max, time.time() + sa_time)
        
        if conflicts == 0 and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
        
        # Try lower m_max
        if best_rows is not None:
            current_m_max = max(bound, best_scope - 1)
        else:
            # No solution yet, try slightly larger
            current_m_max = min(bound + 30, current_m_max + 5)
    
    # Final polish: try to reduce scope further with targeted search
    if best_rows is not None and time.time() < deadline:
        # Try to reduce the maximum element
        max_elem = best_scope
        for _ in range(50):
            if time.time() > deadline:
                break
            # Try to find a row with the max element and reduce it
            target_row = None
            for i, row in enumerate(best_rows):
                if row[-1] == max_elem:
                    target_row = i
                    break
            if target_row is None:
                break
            
            # Try to reduce the last element
            row = best_rows[target_row]
            for new_last in range(row[-1] - 1, row[-2] if len(row) > 1 else 0, -1):
                row[-1] = new_last
                if is_valid(best_rows):
                    _, new_scope = get_conflicts_and_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        max_elem = new_scope
                        break
                row[-1] = max_elem  # revert
            else:
                break
    
    # Fallback: if we never found a valid solution, use greedy with larger M
    if best_rows is None:
        for M in range(bound + 31, bound + 200):
            if time.time() > deadline:
                break
            rows = greedy_construct(n_rows, k, M)
            if rows is not None:
                _, scope = get_conflicts_and_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                break
    
    if best_rows is None:
        # Absolute fallback: just return something valid (may have large scope)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows
