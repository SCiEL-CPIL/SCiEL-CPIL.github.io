"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def get_differences(row):
        """Get all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_row(row, used_diffs):
        """Check if row can be added without conflicting with used_diffs."""
        row_diffs = get_differences(row)
        return not (row_diffs & used_diffs)
    
    def greedy_construct(n_rows, k, rng=None):
        """Greedy construction: pick smallest valid x for each new element."""
        if rng is None:
            rng = random.Random()
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, max_iters=5000, rng=None):
        """Try to reduce the maximum value by local moves."""
        if rng is None:
            rng = random.Random()
        
        def get_used_diffs(rows):
            used = set()
            for row in rows:
                used |= get_differences(row)
            return used
        
        def get_max_val(rows):
            return max(max(row) for row in rows)
        
        def try_reduce_row(row, used_diffs_except_row, target_max):
            """Try to find a better row with max value <= target_max."""
            n = len(row)
            # Try to find a new row with max value less than current
            best_row = None
            # Sample some random rows
            for _ in range(100):
                # Generate a random increasing sequence starting at 0
                # with max value <= target_max
                candidates = list(range(1, target_max + 1))
                if len(candidates) < n - 1:
                    continue
                chosen = [0] + sorted(rng.sample(candidates, n - 1))
                if chosen[-1] >= target_max:
                    continue
                # Check validity
                new_diffs = get_differences(chosen)
                if new_diffs & used_diffs_except_row:
                    continue
                if best_row is None or chosen[-1] < best_row[-1]:
                    best_row = chosen
            return best_row
        
        current_max = get_max_val(rows)
        best_rows = [list(r) for r in rows]
        best_max = current_max
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random row to try to improve
            idx = rng.randint(0, n_rows - 1)
            used_except = get_used_diffs(rows)
            # Remove the differences of the row we're trying to change
            row_diffs = get_differences(rows[idx])
            used_except -= row_diffs
            
            # Try to find a row with smaller max
            target_max = rows[idx][-1] - 1
            if target_max < k + 1:  # minimum possible max is k+1 (0,1,2,...,k+1)
                continue
            
            new_row = try_reduce_row(rows[idx], used_except, target_max)
            if new_row is not None:
                rows[idx] = new_row
                new_max = get_max_val(rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
        
        return best_rows, best_max
    
    # Initial greedy solution
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Try multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomized greedy: add some randomness
        rng = random.Random(restart * 12345 + 67890)
        used = set()
        rows = []
        valid = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                found = False
                while x < best_max + 50:  # don't go too far
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    valid = False
                    break
            rows.append(row)
        
        if not valid:
            continue
        
        current_max = max(max(row) for row in rows)
        if current_max >= best_max:
            continue
        
        # Local search to improve
        improved_rows, improved_max = local_search(rows, n_rows, k, max_iters=2000, rng=rng)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
        
        if best_max <= 112:  # best known
            break
    
    return best_rows
# EVOLVE-BLOCK-END
