"""Improved program for difference triangle set (7,5)."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _validate(rows, n_rows, k):
    """Check validity: all rows start at 0, strictly increasing, length k+1,
    and all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    if len(rows) != n_rows:
        return False
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _construct_greedy(n_rows, k, deadline=None):
    """Greedy construction: each row takes smallest marks whose differences are unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
        if deadline is not None and time.time() > deadline:
            break
    return rows


def _construct_randomized(n_rows, k, seed, deadline=None):
    """Randomized greedy: at each step, pick a random valid mark from a window."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # Collect a few candidates and pick randomly from them
                candidates = [x]
                for trial in range(5):
                    x += 1
                    nd = {x - a for a in row}
                    if not (nd & used):
                        candidates.append(x)
                chosen = rng.choice(candidates)
                # Reconstruct: we need to set x properly
                # Actually let's just pick the first one for simplicity in this variant
                # Better: pick randomly from candidates
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                # Reset x to chosen for next iteration
                x = chosen
        rows.append(row)
        if deadline is not None and time.time() > deadline:
            break
    return rows


def _local_search(rows, n_rows, k, deadline, rng):
    """Try to improve via local search: swap/replace marks to reduce max entry."""
    best_rows = [list(r) for r in rows]
    best_score = _max_entry(best_rows)
    
    used = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    while time.time() < deadline:
        # Pick a random row and a random position to try changing
        row_idx = rng.randint(0, len(best_rows) - 1)
        row = best_rows[row_idx]
        if len(row) < k + 1:
            break
        
        # Try to reduce the last element or a random element
        pos = rng.randint(1, len(row) - 1)  # Don't change position 0 (always 0)
        
        # Try reducing the value at pos
        current_val = row[pos]
        improved = False
        for new_val in range(current_val - 1, 0, -1):
            # Check if row remains strictly increasing
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute new differences for this row
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            
            # Remove old diffs, add new ones
            new_row = row[:]
            new_row[pos] = new_val
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            # Check: new_diffs minus old_diffs should not conflict with other rows' diffs
            other_diffs = used - old_diffs
            if not (new_diffs & other_diffs):
                # Valid move
                used = other_diffs | new_diffs
                best_rows[row_idx] = new_row
                new_score = _max_entry(best_rows)
                if new_score < best_score:
                    best_score = new_score
                improved = True
                break
        
        if not improved:
            # Try increasing a value slightly to allow other improvements
            # Or just continue
            pass
    
    return best_rows, best_score


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with greedy
    rows = _construct_greedy(n_rows, k, deadline)
    if not rows:
        return []
    
    best_rows = rows
    best_score = _max_entry(best_rows)
    
    # Try multiple randomized constructions
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        candidate = _construct_randomized(n_rows, k, seed=42 + attempt, deadline=deadline)
        if candidate and len(candidate) == n_rows:
            if all(len(r) == k + 1 for r in candidate):
                score = _max_entry(candidate)
                if score < best_score:
                    best_rows = candidate
                    best_score = score
        
        # Also try a variant: greedy but with different row ordering
        if attempt % 3 == 0:
            # Try constructing rows in random order
            used = set()
            rand_rows = [None] * n_rows
            order = list(range(n_rows))
            rng.shuffle(order)
            valid = True
            for idx in order:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                rand_rows[idx] = row
            
            if all(r is not None for r in rand_rows):
                score = _max_entry(rand_rows)
                if score < best_score:
                    best_rows = rand_rows
                    best_score = score
    
    # Local search refinement
    if time.time() < deadline - 2.0:
        refined, refined_score = _local_search(
            best_rows, n_rows, k, deadline - 1.0, rng
        )
        if refined_score < best_score:
            best_rows = refined
            best_score = refined_score
    
    # Final validation
    if not _validate(best_rows, n_rows, k):
        # Fallback to pure greedy
        best_rows = _construct_greedy(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
