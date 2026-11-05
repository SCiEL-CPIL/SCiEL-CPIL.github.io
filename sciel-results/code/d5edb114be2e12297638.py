import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses Simulated Annealing / Local Search approach.
    """
    start_time = time.time()
    max_time = 115.0
    
    best_rows = None
    best_scope = float('inf')
    
    def get_conflicts_and_scope(rows):
        """
        Returns (num_conflicts, max_scope).
        num_conflicts is the number of difference values that appear more than once.
        """
        diff_counts = {}
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diff_counts:
                        diff_counts[d] += 1
                    else:
                        diff_counts[d] = 1
        
        conflicts = 0
        for d, c in diff_counts.items():
            if c > 1:
                conflicts += 1
        return conflicts, max_val

    def local_search(rows, rng):
        """
        Attempts to improve the given rows by reducing conflicts first, then scope.
        """
        current_conflicts, current_scope = get_conflicts_and_scope(rows)
        no_improve_count = 0
        max_iters = 5000
        
        for _ in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            # Pick a random row and a random position (1 to k)
            if current_conflicts == 0:
                # Bias towards reducing the global max
                max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
                # With 50% chance, pick the max element of the max row
                if rng.random() < 0.5:
                    r_idx = max_row_idx
                    j_idx = k # The last element is the max
                else:
                    r_idx = rng.randint(0, n_rows - 1)
                    j_idx = rng.randint(1, k)
            else:
                r_idx = rng.randint(0, n_rows - 1)
                j_idx = rng.randint(1, k)
            
            row = rows[r_idx]
            old_val = row[j_idx]
            
            # Determine valid range for new value
            min_val = row[j_idx - 1] + 1
            max_val = row[j_idx + 1] - 1 if j_idx < k else current_scope + 50
            
            # Heuristic for new value selection
            if current_conflicts > 0:
                # Search in a wide range to escape conflicts
                if max_val > min_val:
                    new_val = rng.randint(min_val, max_val)
                else:
                    continue
            else:
                # Try to reduce scope. 
                # If we are moving the last element (max), we prefer smaller values.
                if j_idx == k:
                    if old_val > min_val:
                        new_val = rng.randint(min_val, old_val - 1)
                    else:
                        continue # Can't reduce min_val
                else:
                    # For non-max elements, we can move up or down, but prefer down if it helps?
                    # Actually, moving non-max elements doesn't directly reduce scope unless it allows max to reduce.
                    # Let's just pick a random valid value.
                    if max_val > min_val:
                        new_val = rng.randint(min_val, max_val)
                    else:
                        continue
            
            if new_val == old_val:
                continue
                
            # Check if new_val is valid (strictly increasing)
            # min_val and max_val already ensure this.
            
            # Calculate delta
            row[j_idx] = new_val
            new_conflicts, new_scope = get_conflicts_and_scope(rows)
            
            # Acceptance criterion
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                elif new_scope == current_scope:
                    # Accept with probability to explore
                    if rng.random() < 0.1:
                        accept = True
            else:
                # Conflicts increased
                pass
            
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                no_improve_count = 0
            else:
                # Revert
                row[j_idx] = old_val
                no_improve_count += 1
                if no_improve_count > 500:
                    break
        
        return rows, current_conflicts, current_scope

    # Multi-start
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        rng = random.Random(attempt * 1000 + 42)
        
        # Initialize rows
        rows = []
        for _ in range(n_rows):
            row = [0]
            cur = 0
            for _ in range(k):
                # Add random gap
                gap = rng.randint(1, 10)
                cur += gap
                row.append(cur)
            rows.append(row)
        
        # Run local search
        rows, conf, scope = local_search(rows, rng)
        
        if conf == 0 and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: Greedy
        used = set()
        rows = []
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
        
    return best_rows
