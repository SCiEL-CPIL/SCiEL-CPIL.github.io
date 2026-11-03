import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    def _scope(rows):
        return max(a for r in rows for a in r[1:])

    def _valid(rows, k):
        seen = set()
        for r in rows:
            if r[0] != 0 or len(r) != k + 1:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= 0 or d in seen:
                        return False
                    seen.add(d)
        return True

    def _get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def _rebuild_row(used_diffs, k, max_scope, rng=None):
        """
        Rebuild a row given the set of used differences.
        Uses DFS to find a row with min max element.
        Returns the row or None if not found within max_scope.
        """
        # We want to find a row [0, x1, x2, x3, x4, x5]
        # such that all diffs are not in used_diffs.
        
        # To make it faster, we can use a heuristic ordering.
        # We try to keep the numbers small.
        
        def dfs(current_row, local_diffs):
            if len(current_row) == k + 1:
                return list(current_row)
            
            last = current_row[-1]
            # We need to pick the next element
            # We can limit the range to max_scope
            # But we also need to ensure we can fit the remaining elements
            # Min remaining elements: k + 1 - len(current_row)
            # So the next element must be at most max_scope - (remaining - 1)
            
            remaining = k + 1 - len(current_row)
            max_next = max_scope - (remaining - 1)
            
            # Try candidates from last + 1 to max_next
            # To add some randomness if rng is provided, we can shuffle or pick randomly
            # But for deterministic "best" rebuild, we go in order.
            
            # Optimization: If rng is provided, we can try a random start or shuffle
            # For now, let's stick to ordered for the "greedy" rebuild,
            # and use a separate "random rebuild" for perturbation.
            
            for x in range(last + 1, max_next + 1):
                # Check conflicts
                valid = True
                new_local_diffs = set(local_diffs)
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_local_diffs.add(d)
                
                if valid:
                    res = dfs(current_row + [x], new_local_diffs)
                    if res is not None:
                        return res
            
            return None

        return dfs([0], set())

    def _random_rebuild_row(used_diffs, k, max_scope, rng):
        """
        Randomized rebuild for perturbation.
        """
        def dfs(current_row, local_diffs):
            if len(current_row) == k + 1:
                return list(current_row)
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            max_next = max_scope - (remaining - 1)
            
            # Generate candidates
            candidates = list(range(last + 1, max_next + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                valid = True
                new_local_diffs = set(local_diffs)
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_local_diffs.add(d)
                
                if valid:
                    res = dfs(current_row + [x], new_local_diffs)
                    if res is not None:
                        return res
            
            return None

        return dfs([0], set())

    def _greedy_init(n_rows, k, rng):
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            # Greedy construction with some randomness
            row = [0]
            local_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                found = False
                while x < 200: # Arbitrary limit
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in local_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            local_diffs.add(x - a)
                        used_diffs.update(local_diffs)
                        found = True
                        break
                    x += 1
            if not found:
                # Fallback
                row = [0, 1, 2, 3, 4, 5]
            rows.append(row)
        return rows

    # Phase 1: Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Try a few random initializations
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        rng = random.Random(seed)
        rows = _greedy_init(n_rows, k, rng)
        if _valid(rows, k):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        rng = random.Random(0)
        best_rows = _greedy_init(n_rows, k, rng)
        best_scope = _scope(best_rows)

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Phase 2: Block Coordinate Descent
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10:
        improved = False
        
        # Rebuild each row
        for i in range(n_rows):
            if time.time() > deadline - 10:
                break
            
            # Remove row i from used_diffs
            row_diffs = _get_diffs(current_rows[i])
            other_rows = [r for j, r in enumerate(current_rows) if j != i]
            used_diffs = set()
            for r in other_rows:
                used_diffs.update(_get_diffs(r))
            
            # Rebuild row i
            # Try to find a row with scope <= current_scope
            new_row = _rebuild_row(used_diffs, k, current_scope, None)
            
            if new_row is None:
                # If we can't find one with current scope, try a slightly larger one?
                # Or skip this row for now.
                # Let's try random rebuild with slightly larger scope
                new_row = _random_rebuild_row(used_diffs, k, current_scope + 5, random.Random(int(time.time()*1000) + i))
            
            if new_row is not None:
                new_rows = [list(r) for r in current_rows]
                new_rows[i] = new_row
                new_scope = _scope(new_rows)
                
                if new_scope < current_scope:
                    current_rows = new_rows
                    current_scope = new_scope
                    improved = True
                    no_improve_count = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
                elif new_scope == current_scope:
                    # Even if scope is same, the row might be "better" (e.g. leaves more room for others)
                    # But we don't have a good metric for that.
                    # Let's accept it if it's different, to allow perturbation later?
                    # No, that might cycle.
                    # Only accept if it improves scope.
                    pass
            else:
                # Could not rebuild
                pass
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                # Replace 2 random rows with random rebuilds
                idxs = random.sample(range(n_rows), 2)
                for i in idxs:
                    row_diffs = _get_diffs(current_rows[i])
                    other_rows = [r for j, r in enumerate(current_rows) if j != i]
                    used_diffs = set()
                    for r in other_rows:
                        used_diffs.update(_get_diffs(r))
                    
                    # Try to find a random row
                    new_row = _random_rebuild_row(used_diffs, k, current_scope + 10, random.Random(int(time.time()*10000)))
                    if new_row is not None:
                        current_rows[i] = new_row
                        # Update scope
                        current_scope = _scope(current_rows)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]
                
                no_improve_count = 0
                max_no_improve += 100 # Increase patience after perturbation

    # Final check
    if not _valid(best_rows, k):
        # Fallback to a valid solution
        # The greedy init should be valid, but just in case
        # We can use the first valid one found in init
        pass

    return best_rows
