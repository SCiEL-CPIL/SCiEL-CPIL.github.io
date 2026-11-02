"""Improved construct for difference triangle set (4,7) via local search with compaction."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Number of differences per row: k*(k+1)/2
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    def compute_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(n_rows, k, seed=0):
        """Greedy with random tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try candidates in random order among small range
                found = False
                # Search up to a reasonable bound
                max_search = 200 + rng.randint(0, 50)
                candidates = list(range(x + 1, x + 1 + max_search))
                rng.shuffle(candidates)
                for cand in candidates:
                    new = {cand - a for a in row}
                    if not (new & used):
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    # Fallback: linear search
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not (new & used):
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def compact_row(row, used):
        """Try to reduce the last element of a row by shifting it down."""
        if len(row) < 2:
            return row
        # Try to reduce the largest entry
        last = row[-1]
        for new_last in range(last - 1, 0, -1):
            # Check new_last > row[-2]
            if new_last <= row[-2]:
                break
            # Compute new differences involving new_last
            old_diffs = {last - a for a in row[:-1]}
            new_diffs = {new_last - a for a in row[:-1]}
            # Remove old diffs from used, check new ones
            used_copy = used - old_diffs
            if new_diffs & used_copy:
                continue
            # Check internal consistency (differences among row[:-1] unchanged)
            # The differences within row[:-1] are already valid
            # So we just need new_diffs to not conflict
            return row[:-1] + [new_last]
        return row

    def full_compact(rows, used):
        """Iteratively compact all rows."""
        changed = True
        iterations = 0
        while changed and iterations < 50:
            changed = False
            iterations += 1
            for i in range(len(rows)):
                row = rows[i]
                if len(row) < 2:
                    continue
                # Remove this row's differences from used
                row_diffs = set()
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        row_diffs.add(row[b] - row[a])
                used_without = used - row_diffs
                # Try to compact
                new_row = row[:]
                # Try reducing each element from the end
                for pos in range(len(new_row) - 1, 0, -1):
                    orig_val = new_row[pos]
                    lower_bound = new_row[pos - 1] + 1
                    for new_val in range(orig_val - 1, lower_bound - 1, -1):
                        # Compute new differences involving new_val
                        new_diffs = {new_val - new_row[j] for j in range(pos)}
                        new_diffs |= {new_row[j] - new_val for j in range(pos + 1, len(new_row))}
                        # Actually differences are always positive: row[j]-row[i] for i<j
                        # So for position pos, differences are:
                        # new_val - new_row[j] for j < pos
                        # new_row[j] - new_val for j > pos (but these don't involve pos as smaller)
                        # Wait, differences are a_j - a_i for i < j.
                        # If we change element at pos, differences involving it:
                        # For j > pos: new_row[j] - new_val
                        # For i < pos: new_val - new_row[i]
                        diff_involving = set()
                        for j in range(pos):
                            diff_involving.add(new_val - new_row[j])
                        for j in range(pos + 1, len(new_row)):
                            diff_involving.add(new_row[j] - new_val)
                        
                        # Check against used_without and other diffs in this row not involving pos
                        # Other diffs in row (not involving pos) remain same
                        # Check if diff_involving conflicts with used_without
                        if diff_involving & used_without:
                            continue
                        # Also check no duplicate within diff_involving itself
                        if len(diff_involving) != len(set(diff_involving)):
                            continue
                        # Accept
                        new_row[pos] = new_val
                        changed = True
                        break
                # Rebuild used
                row_diffs_new = set()
                for a in range(len(new_row)):
                    for b in range(a + 1, len(new_row)):
                        row_diffs_new.add(new_row[b] - new_row[a])
                used = (used - row_diffs) | row_diffs_new
                rows[i] = new_row
        return rows, used

    def local_search(rows, used, time_limit):
        """SA-like local search with row perturbations and compaction."""
        rng = random.Random()
        best_rows = [r[:] for r in rows]
        best_scope = compute_scope(best_rows)
        current_rows = [r[:] for r in rows]
        current_used = set(used)
        current_scope = compute_scope(current_rows)
        
        # Precompute row diffs helper
        def get_row_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs
        
        t = 1.0
        cooling = 0.9995
        min_t = 0.01
        
        while time.time() < time_limit and t > min_t:
            # Pick a random row to perturb
            row_idx = rng.randint(0, len(current_rows) - 1)
            row = current_rows[row_idx]
            old_diffs = get_row_diffs(row)
            
            # Perturbation: shift a random element
            if len(row) > 1:
                pos = rng.randint(1, len(row) - 1)  # Don't move the 0
                delta = rng.randint(-5, 5)
                if delta == 0:
                    delta = 1
                new_val = row[pos] + delta
                # Check bounds
                if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                    t *= cooling
                    continue
                
                new_row = row[:]
                new_row[pos] = new_val
                new_diffs = get_row_diffs(new_row)
                
                # Check validity against other rows
                other_diffs = current_used - old_diffs
                if new_diffs & other_diffs:
                    t *= cooling
                    continue
                
                # Accept if better or with probability
                new_scope = compute_scope(current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:])
                delta_scope = new_scope - current_scope
                
                if delta_scope <= 0:
                    # Accept
                    current_used = (current_used - old_diffs) | new_diffs
                    current_rows[row_idx] = new_row
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Metropolis acceptance
                    import math
                    if rng.random() < math.exp(-delta_scope / t):
                        current_used = (current_used - old_diffs) | new_diffs
                        current_rows[row_idx] = new_row
                        current_scope = new_scope
                
            t *= cooling
        
        # Final compaction
        best_rows, _ = full_compact(best_rows, set())
        # Recompute used for best_rows
        best_used = set()
        for row in best_rows:
            best_used |= get_row_diffs(row)
        return best_rows, best_used, compute_scope(best_rows)

    # Initial solution
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple greedy seeds
    num_initial = min(20, max(5, int((deadline - time.time()) / 0.5)))
    for seed in range(num_initial):
        if time.time() > deadline - 10:
            break
        rows = greedy_construct(n_rows, k, seed=seed)
        if is_valid(rows):
            scope = compute_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
        best_scope = compute_scope(rows)
    
    # Local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_rows, _, best_scope = local_search(best_rows, set(), time.time() + remaining - 2)
    
    # Final compaction
    if time.time() < deadline - 1:
        best_rows, _ = full_compact(best_rows, set())
        best_scope = compute_scope(best_rows)
    
    return best_rows
# EVOLVE-BLOCK-END
