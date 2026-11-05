import time
import random
import math
import sys


def _row_diffs(row):
    """Return list of all positive differences within a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            m |= (1 << (row[j] - row[i]))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _generate_candidate_rows(max_scope, num_elements, num_candidates=200):
    """
    Generate many candidate rows (sets of num_elements increasing ints starting at 0,
    all <= max_scope) using random construction. Each row must have internally distinct diffs.
    Returns a list of (row, mask, scope).
    """
    candidates = []
    seen_masks = set()
    
    for _ in range(num_candidates):
        # Random row: pick num_elements-1 values from 1..max_scope
        if max_scope < num_elements - 1:
            continue
        vals = sorted(random.sample(range(1, max_scope + 1), num_elements - 1))
        row = [0] + vals
        # Check internal distinctness
        diffs = set()
        valid = True
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        if not valid:
            continue
        mask = 0
        for d in diffs:
            mask |= (1 << d)
        if mask in seen_masks:
            continue
        seen_masks.add(mask)
        candidates.append((row, mask, row[-1]))
    
    return candidates


def _greedy_row_assignment(candidates, n_rows, used_mask):
    """
    Greedily assign n_rows from candidates, each with scope <= max_scope,
    such that no two rows share a difference. Returns list of rows or None.
    """
    # Sort candidates by scope ascending (prefer small rows)
    cands = sorted(candidates, key=lambda x: x[2])
    
    rows = []
    local_mask = 0
    
    for row_idx in range(n_rows):
        found = False
        for row, mask, scope in cands:
            if (mask & (used_mask | local_mask)) == 0:
                rows.append(row)
                local_mask |= mask
                found = True
                break
        if not found:
            return None
    
    return rows


def _backtrack_global(candidates, n_rows, used_mask, max_scope, deadline, rng):
    """
    Backtrack over rows: for each row position, try candidates in order of scarcity.
    Uses forward checking: a candidate is viable if its mask doesn't conflict with used.
    Orders candidates by number of remaining compatible options (scarcity heuristic).
    """
    start = time.time()
    
    # Pre-filter: only candidates whose mask doesn't conflict with used_mask
    available = [(row, mask, scope) for row, mask, scope in candidates
                 if (mask & used_mask) == 0]
    
    if len(available) < n_rows:
        return None
    
    # Sort by scope, then by mask sparsity (fewer bits set = more flexible)
    available.sort(key=lambda x: (x[2], bin(x[1]).count('1')))
    
    def backtrack(idx, current_mask, current_rows):
        if time.time() > deadline:
            return None
        if idx == n_rows:
            return current_rows[:]
        
        # Scarcity: count how many candidates are still compatible
        # Try candidates that leave the most flexibility for remaining rows
        # Heuristic: prefer candidates whose mask conflicts with the fewest other available candidates
        best_candidate = None
        best_conflict_count = float('inf')
        
        # Sample a subset for speed
        if len(available) > 50:
            sample_size = min(50, len(available))
            indices = list(range(len(available)))
            rng.shuffle(indices)
            subset = [available[i] for i in indices[:sample_size]]
        else:
            subset = available
        
        for row, mask, scope in subset:
            if (mask & (used_mask | current_mask)) != 0:
                continue
            
            # Estimate conflicts: how many other available candidates does this mask block?
            # Cheap proxy: just use scope as tiebreaker, try in order
            # For speed, just try in sorted order
            new_rows = current_rows + [row]
            result = backtrack(idx + 1, current_mask | mask, new_rows)
            if result is not None:
                return result
        
        return None
    
    return backtrack(0, 0, [])


def _construct_with_backtracking(n_rows, num_elements, deadline, rng):
    """
    Main strategy: generate candidate rows, then use global backtracking to assign
    n_rows rows with no shared differences. Try increasing max_scope.
    """
    counting_bound = math.ceil(math.sqrt(2 * n_rows * num_elements * (num_elements - 1) / 2))
    
    # Start from counting bound, increase
    for max_scope in range(counting_bound, counting_bound + 60):
        if time.time() > deadline:
            break
        
        # Generate candidates for this scope
        num_cands = max(300, 100 * n_rows)
        candidates = _generate_candidate_rows(max_scope, num_elements, num_cands, rng)
        
        if len(candidates) < n_rows:
            continue
        
        # Try backtracking
        result = _backtrack_global(candidates, n_rows, 0, max_scope, deadline, rng)
        if result is not None:
            return result, max_scope
    
    return None, None


def _local_search_refine(rows, num_elements, deadline, rng):
    """
    After global construction, try to reduce scope by re-optimizing individual rows.
    For each row, try to find a replacement with smaller scope.
    """
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    n_rows = len(current_rows)
    
    def compute_forbidden_except(exclude_idx):
        mask = 0
        for i in range(n_rows):
            if i != exclude_idx:
                mask |= _row_mask(current_rows[i])
        return mask
    
    improved = True
    max_rounds = 10
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            forbidden = compute_forbidden_except(row_idx)
            old_row = current_rows[row_idx]
            old_scope_val = old_row[-1]
            
            # Try to find a smaller-scope row
            for new_scope in range(num_elements - 1, old_scope_val):
                if time.time() > deadline:
                    break
                
                # Generate candidates with this scope
                cands = _generate_candidate_rows(new_scope, num_elements, 100, rng)
                found = False
                for row, mask, scope in cands:
                    if (mask & forbidden) == 0:
                        current_rows[row_idx] = row
                        new_total = _scope(current_rows)
                        if new_total < best_scope:
                            best_scope = new_total
                            best_rows = [r[:] for r in current_rows]
                        improved = True
                        found = True
                        break
                if found:
                    break
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 130.0  # safety margin
    
    num_elements = k + 1
    counting_bound = math.ceil(math.sqrt(2 * n_rows * num_elements * (num_elements - 1) / 2))
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Global backtracking construction with multiple random seeds
    for seed in range(30):
        if time.time() > DEADLINE - 40:
            break
        rng_local = random.Random(seed * 7919 + 13)
        
        rows, scope = _construct_with_backtracking(
            n_rows, num_elements, DEADLINE - 30, rng_local
        )
        
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    
    # Fallback: simple sequential greedy if backtracking failed
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search refinement
    if time.time() < DEADLINE - 15:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, DEADLINE - 10, random.Random(99)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if best_rows and _is_valid(best_rows):
        return best_rows
    else:
        # Absolute fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
