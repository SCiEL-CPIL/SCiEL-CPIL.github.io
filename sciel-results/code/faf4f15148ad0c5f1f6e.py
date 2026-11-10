import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    total_diffs_per_row = k * (k + 1) // 2
    counting_lb = n_rows * total_diffs_per_row

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def score_of(rows):
        return max(max(r) for r in rows)

    # --- Phase 1: Greedy initial solution ---
    def greedy_construct(max_val):
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                found = False
                for v in range(base + 1, max_val + 1):
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        row.append(v)
                        for a in row[:-1]:
                            local.add(v - a)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
            used |= row_diffs(row)
        return rows

    hint = max(counting_lb + 20, 130)
    init_rows = greedy_construct(hint)
    if init_rows is None:
        hint = counting_lb + 80
        init_rows = greedy_construct(hint)
    if init_rows is None:
        # Unbounded greedy
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)

    # --- Phase 2: Row-by-row backtracking with forward checking ---
    # We build rows one at a time. At each step, we pick the next value for
    # the current row, checking that all new differences are unused globally.
    # We use randomized restarts with different orderings to escape local minima.

    def backtrack_build(max_val, rng_seed=0):
        """Build n_rows rows with all entries <= max_val using backtracking.
        Returns list of rows or None."""
        rng = random.Random(rng_seed)
        rows = []
        used = set()
        # Track which difference conflicts with which row for backjumping
        # conflict_map: diff -> row_index that used it
        
        def build_row(row_idx, max_val):
            """Build row row_idx. Returns (row, new_diffs) or None."""
            row = [0]
            local = set()
            
            def rec(pos):
                if time.time() > deadline:
                    return None
                if pos == k:
                    return row[:]
                base = row[-1]
                # Generate candidate values in random-ish order for diversity
                cands = []
                for v in range(base + 1, max_val + 1):
                    # Check local and global conflicts
                    conflict = False
                    for a in row:
                        d = v - a
                        if d in local or d in used:
                            conflict = True
                            break
                    if not conflict:
                        cands.append(v)
                    if len(cands) > 15:
                        break
                
                if not cands:
                    return None
                
                # Try candidates in shuffled order
                rng.shuffle(cands)
                for v in cands:
                    # Check all differences
                    new_diffs = set()
                    ok = True
                    for a in row:
                        d = v - a
                        if d in local or d in used:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    
                    row.append(v)
                    local.update(new_diffs)
                    result = rec(pos + 1)
                    if result is not None:
                        return result
                    row.pop()
                    local.difference_update(new_diffs)
                
                return None
            
            return rec(0)
        
        for i in range(n_rows):
            if time.time() > deadline:
                return None
            row = build_row(i, max_val)
            if row is None:
                return None
            rows.append(row)
            used |= row_diffs(row)
        
        return rows

    # Iterative deepening: try to find a solution with max_val < best_score
    for target in range(best_score - 1, max(counting_lb - 1, 0), -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random seeds to increase diversity
        found = False
        for seed in range(50):
            if time.time() > deadline - 3:
                break
            sol = backtrack_build(target, seed)
            if sol is not None:
                if is_valid(sol):
                    new_score = score_of(sol)
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [r[:] for r in sol]
                    found = True
                    break
        
        if found:
            # If we found a solution at this target, try to go lower
            continue
        else:
            # No solution at this target, try next lower
            continue

    if not is_valid(best_rows):
        return init_rows

    return best_rows

# EVOLVE-BLOCK-END
