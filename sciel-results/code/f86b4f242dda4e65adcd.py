"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _greedy_row(used, k, max_x, rng):
    """Build one row of k+1 increasing ints starting at 0, using greedy with randomization.
    
    used: set of already-used positive differences (global across rows).
    k: number of gaps, so row has k+1 entries.
    max_x: upper bound for entries.
    rng: random.Random instance.
    Returns (row, new_used_diffs) or (None, None) if stuck.
    """
    row = [0]
    local_used = set()
    for _ in range(k):
        # Find all valid next values
        candidates = []
        x = row[-1] + 1
        while x <= max_x:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & local_used):
                candidates.append((x, new_diffs))
            x += 1
        if not candidates:
            return None, None
        # Pick randomly from candidates, biased toward smaller values
        # Use weighted random: prefer smaller x
        if len(candidates) <= 8:
            idx = rng.randint(0, len(candidates) - 1)
        else:
            # Exponential decay preference for smaller values
            weights = [1.0 / (1.0 + c[0] * 0.1) for c in candidates]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    idx = i
                    break
            else:
                idx = len(candidates) - 1
        x, new_diffs = candidates[idx]
        row.append(x)
        local_used |= new_diffs
    used.update(local_used)
    return row, local_used


def _score(rows):
    """Return the largest entry (scope) of a valid set of rows."""
    if not rows:
        return float('inf')
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check that all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope)."""
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Initial greedy construction to get a baseline
    rng = random.Random(42)
    
    # First, try deterministic greedy for a quick baseline
    used = set()
    baseline_rows = []
    for _ in range(n_rows):
        row, _ = _greedy_row(used, k, 500, rng)
        if row is None:
            break
        baseline_rows.append(row)
    
    if len(baseline_rows) < n_rows:
        # Fallback: very simple greedy
        used = set()
        baseline_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            baseline_rows.append(row)
    
    best_rows = [list(r) for r in baseline_rows]
    best_scope = _score(best_rows)
    
    # Iterative improvement: local search by rebuilding rows
    # Strategy: repeatedly pick a random row to rebuild, trying to reduce scope
    
    max_x_limit = best_scope + 50  # allow some flexibility
    
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Strategy 1: Try to reduce scope by rebuilding the row with the largest max
        # Find the row with the largest maximum
        max_indices = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        
        # Try rebuilding the worst row
        target_idx = max_indices[0]
        old_row = best_rows[target_idx]
        old_max = old_row[-1]
        
        # Remove old row's differences from used
        used = set()
        for i, row in enumerate(best_rows):
            if i == target_idx:
                continue
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    used.add(row[b] - row[a])
        
        # Try to build a new row with smaller max
        new_row, new_diffs = _greedy_row(used, k, old_max - 1, rng)
        if new_row is not None:
            new_scope = max(best_rows[i][-1] for i in range(n_rows) if i != target_idx)
            new_scope = max(new_scope, new_row[-1])
            if new_scope < best_scope:
                best_rows[target_idx] = new_row
                best_scope = new_scope
                max_x_limit = best_scope + 50
                continue
        
        # Strategy 2: Random local search - swap/rebuild random row
        if rng.random() < 0.3:
            target_idx = rng.randint(0, n_rows - 1)
            old_row = best_rows[target_idx]
            
            used = set()
            for i, row in enumerate(best_rows):
                if i == target_idx:
                    continue
                n = len(row)
                for a in range(n):
                    for b in range(a + 1, n):
                        used.add(row[b] - row[a])
            
            # Try building with a random offset or different strategy
            for attempt in range(3):
                new_row, new_diffs = _greedy_row(used, k, best_scope, rng)
                if new_row is not None:
                    new_scope = max(best_rows[i][-1] for i in range(n_rows) if i != target_idx)
                    new_scope = max(new_scope, new_row[-1])
                    if new_scope < best_scope:
                        best_rows[target_idx] = new_row
                        best_scope = new_scope
                        break
        
        # Strategy 3: Try to compress by rebuilding multiple rows
        if iteration % 50 == 0 and time.time() < deadline - 10.0:
            # Try a full rebuild with different random seed
            used = set()
            candidate_rows = []
            success = True
            for _ in range(n_rows):
                row, _ = _greedy_row(used, k, best_scope, rng)
                if row is None:
                    success = False
                    break
                candidate_rows.append(row)
            if success:
                new_scope = _score(candidate_rows)
                if new_scope < best_scope:
                    best_rows = [list(r) for r in candidate_rows]
                    best_scope = new_scope
    
    return best_rows
# EVOLVE-BLOCK-END
