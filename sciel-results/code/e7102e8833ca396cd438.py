"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set (n_rows, k) minimizing the largest entry.
    Uses greedy construction with randomization + local search (simulated annealing).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Number of differences per row: k*(k+1)/2
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Simple counting bound lower bound on scope:
    # We need total_diffs distinct positive differences, max diff <= scope
    # So scope must satisfy scope*(scope+1)/2 >= total_diffs
    lb_scope = int(np.ceil((-1 + np.sqrt(1 + 8 * total_diffs)) / 2))
    
    def row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            rd = row_diffs(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    def count_conflicts(rows):
        """Count number of conflicting differences (differences appearing in >1 row)."""
        from collections import Counter
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        c = Counter(all_diffs)
        return sum(v - 1 for v in c.values() if v > 1)
    
    def greedy_construct(n_rows, k, max_scope, rng):
        """Greedy construction: build rows one by one, picking smallest valid next mark."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_scope:
                    return None
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def random_construct(n_rows, k, max_scope, rng):
        """Random construction: pick random increasing sequences within [0, max_scope]."""
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            # Randomly pick k more values from 1..max_scope
            candidates = list(range(1, max_scope + 1))
            rng.shuffle(candidates)
            # Try to build a valid row
            for cand in candidates:
                if len(row) == k + 1:
                    break
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
            if len(row) != k + 1:
                return None
            used |= row_diffs(row)
        return rows
    
    def local_search(rows, max_scope, rng, time_limit=2.0):
        """
        Local search / simulated annealing to reduce scope.
        Try to move individual entries to reduce the maximum entry.
        """
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in best_rows)
        current_scope = best_scope
        start_ls = time.time()
        
        # Temperature for SA
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() - start_ls < time_limit:
            # Pick a random row and a random non-zero entry to move
            r_idx = rng.randint(0, n_rows - 1)
            # Pick a position (not the first which is 0)
            pos = rng.randint(1, k)
            
            old_val = best_rows[r_idx][pos]
            # Try moving to a new value
            # Bias towards smaller values to reduce scope
            if rng.random() < 0.7:
                # Try a smaller value
                new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try a random value
                new_val = rng.randint(1, max_scope)
            
            if new_val == old_val:
                continue
            
            # Check if the row would still be strictly increasing
            row = best_rows[r_idx]
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
            
            # Compute conflict count for this row with new value
            # We need to check against all other rows' differences
            # For efficiency, compute the diffs of this row with new value
            test_row = row[:]
            test_row[pos] = new_val
            
            # Check strict increase
            for i in range(len(test_row) - 1):
                if test_row[i + 1] <= test_row[i]:
                    continue
            
            # Compute new diffs for this row
            new_rd = set()
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    new_rd.add(test_row[j] - test_row[i])
            
            # Check conflicts with other rows
            conflict = 0
            for i in range(n_rows):
                if i == r_idx:
                    continue
                other_rd = row_diffs(best_rows[i])
                conflict += len(new_rd & other_rd)
            
            # Also check internal consistency (shouldn't have dupes within a row since strictly increasing)
            
            # Acceptance
            new_scope = max(max(r) for r in best_rows if r != test_row)
            # Actually we need to recompute properly
            # For simplicity, just check if this reduces conflicts
            if conflict == 0:
                # Valid move
                best_rows[r_idx] = test_row
                best_scope = max(max(r) for r in best_rows)
                T = max(T_min, T * cooling)
            else:
                # Reject for now (could add SA acceptance)
                T = max(T_min, T * cooling)
        
        return best_rows, best_scope
    
    # Phase 1: Greedy construction to get a valid starting point
    # Find minimal scope via greedy
    best_rows = None
    best_scope = max_scope = lb_scope + n_rows * k  # initial upper bound
    
    # Try greedy with increasing scope
    scope = lb_scope
    while scope <= best_scope and time.time() < deadline:
        rows = greedy_construct(n_rows, k, scope, random)
        if rows is not None:
            best_rows = rows
            best_scope = scope
            break
        scope += 1
    
    if best_rows is None:
        # Fallback: use the original greedy without scope limit
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)
    
    # Phase 2: Local search to try to reduce scope
    if time.time() < deadline - 5:
        improved_rows, improved_scope = local_search(
            best_rows, best_scope, random, 
            time_limit=min(30, max(5, deadline - time.time() - 5))
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    # Phase 3: Multi-start with random construction + local search
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try random construction with current best scope
            rows = random_construct(n_rows, k, best_scope, rng)
            if rows is not None:
                scope = max(max(r) for r in rows)
                # Local search
                ls_rows, ls_scope = local_search(rows, scope, rng, time_limit=1.0)
                if ls_scope < best_scope:
                    best_rows = ls_rows
                    best_scope = ls_scope
    
    # Final validation
    if not is_valid(best_rows):
        # Shouldn't happen, but fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
# EVOLVE-BLOCK-END
