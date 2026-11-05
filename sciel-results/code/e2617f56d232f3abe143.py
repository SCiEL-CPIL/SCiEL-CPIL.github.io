"""Difference triangle set (4,7) - optimized construction."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max entry: need to fit total_diffs differences in [1, M]
    # Each difference d can appear at most once, so M >= total_diffs
    # But also structural constraints; use a reasonable lower bound
    lb = total_diffs  # simple counting bound
    
    best_result = None
    best_score = float('inf')
    
    def validate(rows):
        """Check that all positive differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def score_of(rows):
        """Return the largest entry (the objective to minimize)."""
        return max(max(row) for row in rows)
    
    def greedy_construct(used_diffs, remaining_rows, current_rows):
        """
        Greedily build remaining rows given already-used differences and existing rows.
        Returns the completed set of rows or None if stuck.
        """
        rows = list(current_rows)
        used = set(used_diffs)
        
        for _ in range(remaining_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                for a in row:
                    new_diffs.add(x - a)
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        return rows
    
    def greedy_with_pressure(used_diffs, remaining_rows, current_rows, pressure_weight=1.0):
        """
        Greedy construction with dynamic difference pressure ordering.
        At each step, try candidates in order of how much they increase the max used diff,
        breaking ties by smallest candidate. This packs differences into lower range.
        """
        rows = list(current_rows)
        used = set(used_diffs)
        max_used = max(used) if used else 0
        
        for _ in range(remaining_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                for a in row:
                    d = x - a
                    new_diffs.add(d)
                
                # Check if any new diff is already used
                if new_diffs & used:
                    continue
                
                # Score this candidate: how much it increases the max
                new_max = max(max_used, max(new_diffs))
                # We want to minimize the increase in max, so prefer candidates
                # where new_max is small. Since we iterate x in order, the first
                # valid candidate with smallest new_max is best.
                # Actually, since we go in order of x, the first valid one has
                # the smallest x, which tends to give smaller new_max.
                # Let's just take the first valid one for speed, but we can
                # look ahead a bit.
                row.append(x)
                used |= new_diffs
                max_used = new_max
            
            rows.append(row)
        
        return rows
    
    def randomized_greedy(used_diffs, remaining_rows, current_rows, rng, lookahead=5):
        """
        Randomized greedy: at each position, consider a window of candidates
        and pick one randomly among those that are valid, weighted toward
        smaller values. This allows escaping local optima.
        """
        rows = list(current_rows)
        used = set(used_diffs)
        
        for _ in range(remaining_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find a window of valid candidates
                candidates = []
                x_check = x
                for _ in range(lookahead * 3):  # search window
                    x_check += 1
                    new_diffs = set()
                    for a in row:
                        new_diffs.add(x_check - a)
                    if not (new_diffs & used):
                        candidates.append(x_check)
                        if len(candidates) >= lookahead:
                            break
                
                if not candidates:
                    # Shouldn't happen, but fallback
                    while True:
                        x_check += 1
                        new_diffs = set()
                        for a in row:
                            new_diffs.add(x_check - a)
                        if not (new_diffs & used):
                            candidates.append(x_check)
                            break
                
                # Pick from candidates with bias toward smaller values
                # Use exponential weighting
                weights = [1.0 / (1 + c * 0.01) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
                
                x = chosen
                new_diffs = set()
                for a in row:
                    new_diffs.add(x - a)
                row.append(x)
                used |= new_diffs
            
            rows.append(row)
        
        return rows
    
    def local_search(rows, rng, max_iterations=200):
        """
        Simple local search: try to reduce the largest entry by swapping
        individual elements with smaller valid ones.
        """
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to reduce each element (except 0) to a smaller valid value
            for ri, row in enumerate(rows):
                for pi in range(1, len(row)):
                    old_val = row[pi]
                    # Try values from old_val - 1 down to row[pi-1] + 1
                    for new_val in range(old_val - 1, row[pi - 1], -1):
                        # Check if new_val creates any duplicate differences
                        new_diffs = set()
                        valid = True
                        for i in range(len(row)):
                            if i == pi:
                                continue
                            d = abs(new_val - row[i])
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        # Check against all other rows' differences and this row's other diffs
                        # Remove old diffs involving position pi
                        old_diffs_to_remove = set()
                        for i in range(len(row)):
                            if i != pi:
                                old_diffs_to_remove.add(abs(old_val - row[i]))
                        
                        # Check if new_diffs intersect with (used - old_diffs_to_remove)
                        if new_diffs & (used - old_diffs_to_remove):
                            continue
                        
                        # Also check that new_diffs doesn't have internal duplicates
                        if len(new_diffs) != len([abs(new_val - row[i]) for i in range(len(row)) if i != pi]):
                            continue
                        
                        # Valid move! Apply it
                        row[pi] = new_val
                        used -= old_diffs_to_remove
                        used |= new_diffs
                        improved = True
                        break
        
        return rows
    
    # Strategy 1: Pure greedy
    result = greedy_construct(set(), n_rows, [])
    if result is not None and validate(result):
        s = score_of(result)
        if s < best_score:
            best_score = s
            best_result = [list(r) for r in result]
    
    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of rows or use different random seeds
        result = randomized_greedy(set(), n_rows, [], rng, lookahead=3)
        if result is not None and validate(result):
            s = score_of(result)
            if s < best_score:
                best_score = s
                best_result = [list(r) for r in result]
        
        # Try local search on the best result so far
        if best_result is not None and time.time() < deadline - 5:
            ls_result = local_search(best_result, rng, max_iterations=50)
            if validate(ls_result):
                s = score_of(ls_result)
                if s < best_score:
                    best_score = s
                    best_result = [list(r) for r in ls_result]
    
    # Strategy 3: Try with different random seeds for more diversity
    for seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        for _ in range(10):
            if time.time() > deadline:
                break
            result = randomized_greedy(set(), n_rows, [], rng2, lookahead=4)
            if result is not None and validate(result):
                s = score_of(result)
                if s < best_score:
                    best_score = s
                    best_result = [list(r) for r in result]
    
    # Final local search pass
    if best_result is not None and time.time() < deadline - 2:
        rng3 = random.Random(123)
        ls_result = local_search(best_result, rng3, max_iterations=100)
        if validate(ls_result):
            s = score_of(ls_result)
            if s < best_score:
                best_score = s
                best_result = [list(r) for r in ls_result]
    
    if best_result is None:
        # Fallback: pure greedy
        result = greedy_construct(set(), n_rows, [])
        best_result = result
        best_score = score_of(result)
    
    return best_result
