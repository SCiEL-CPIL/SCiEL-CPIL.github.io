"""Difference triangle set (4,7): construct rows with distinct differences, minimize scope."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def all_diffs_valid(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        """Largest entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_row(used_diffs, k, rng, randomize=False):
        """Build one row greedily, optionally with randomization."""
        row = [0]
        while len(row) < k + 1:
            if randomize:
                # Try a random window of candidate positions
                base = row[-1]
                candidates = []
                for offset in range(1, 60):
                    x = base + offset
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    if len(candidates) > 15:
                        break
                if not candidates:
                    # Fallback: scan further
                    x = base + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        x += 1
                x = rng.choice(candidates)
            else:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
            row.append(x)
            used_diffs |= {x - a for a in row[:-1]}
        return row

    def local_search(rows, rng, max_iters=5000):
        """Improve rows by swapping marks to reduce scope."""
        best_scope = scope(rows)
        best_rows = [list(r) for r in rows]
        current_scope = best_scope
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random row and a random position to try moving
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) <= 1:
                continue
            pi = rng.randrange(1, len(row))  # don't move position 0
            
            old_val = row[pi]
            
            # Try to find a better position (smaller or larger)
            # First try smaller values (to reduce scope)
            found = False
            for new_val in range(old_val - 1, -1, -1):
                if new_val in row:
                    continue
                # Check increasing
                if pi > 0 and new_val <= row[pi - 1]:
                    break
                if pi < len(row) - 1 and new_val >= row[pi + 1]:
                    continue
                # Check differences
                row[pi] = new_val
                if all_diffs_valid(best_rows):
                    s = scope(best_rows)
                    if s < best_scope:
                        best_scope = s
                        found = True
                    else:
                        row[pi] = old_val
                    break
                row[pi] = old_val
            
            if found:
                current_scope = best_scope
                continue
            
            # Try larger values (might enable other improvements)
            for new_val in range(old_val + 1, old_val + 20):
                if new_val in row:
                    continue
                if pi < len(row) - 1 and new_val >= row[pi + 1]:
                    break
                row[pi] = new_val
                if all_diffs_valid(best_rows):
                    s = scope(best_rows)
                    if s < best_scope:
                        best_scope = s
                        found = True
                    else:
                        row[pi] = old_val
                    break
                row[pi] = old_val
            
            if found:
                current_scope = best_scope
        
        return best_rows, best_scope

    def full_greedy(k, rng, randomize=False):
        """Build all rows greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, rng, randomize)
            rows.append(row)
        return rows

    # Strategy: try multiple randomized greedy constructions, then local search
    best_rows = None
    best_scope = float('inf')
    
    # First, deterministic greedy as baseline
    rng = random.Random(42)
    rows = full_greedy(k, rng, randomize=False)
    if all_diffs_valid(rows):
        s = scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = s
    
    # Try multiple randomized constructions
    num_attempts = 0
    while time.time() < deadline - 10.0:
        num_attempts += 1
        seed = 1000 + num_attempts * 7919
        rng = random.Random(seed)
        rows = full_greedy(k, rng, randomize=True)
        if not all_diffs_valid(rows):
            continue
        s = scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]
        
        # Local search on this solution
        if time.time() < deadline - 10.0:
            improved_rows, improved_scope = local_search(rows, rng, max_iters=2000)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = [list(r) for r in improved_rows]
        
        # Also do local search on the current best
        if time.time() < deadline - 10.0:
            improved_rows, improved_scope = local_search(best_rows, rng, max_iters=3000)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = [list(r) for r in improved_rows]
    
    # Final local search pass with more iterations if time permits
    if best_rows is not None and time.time() < deadline - 5.0:
        rng = random.Random(99999)
        improved_rows, improved_scope = local_search(best_rows, rng, max_iters=5000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = [list(r) for r in improved_rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        rng = random.Random(0)
        best_rows = full_greedy(k, rng, randomize=False)
    
    return best_rows
# EVOLVE-BLOCK-END
