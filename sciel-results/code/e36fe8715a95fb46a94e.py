import time
import random
import math
import sys

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1
    row_len = k + 1

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    def obj(rows):
        return scope_of(rows) * 1000000 + sum_of(rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- Deterministic Greedy (Baseline) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- Randomized Greedy ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < window and scanned < 2000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                if rng.random() < bias:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.6))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- Backtracking Row Constructor (Inspiration) ----
    # Finds a row of length k+1 starting at 0, with max element <= max_scope,
    # avoiding differences in 'forbidden'. Minimizes the max element.
    def backtrack_row(forbidden, max_scope, rng, timeout=0.5):
        """
        Returns a valid row or None.
        Uses DFS with pruning.
        """
        start_t = time.time()
        # We want to find *any* valid row with max <= max_scope.
        # To minimize max, we could search for max = 1, 2, ... but that's slow.
        # Instead, we search for any valid row within max_scope.
        # The "minimizing" aspect is handled by the caller trying smaller scopes.
        
        row = [0]
        used_local = set()
        
        # Heuristic: order of candidates matters.
        # We try to pick small values first to keep the row "compact".
        
        def dfs(idx, current_max):
            if time.time() - start_t > timeout:
                return False
            
            if idx == row_len:
                return True
            
            # Pruning: if current_max > max_scope, fail
            if current_max > max_scope:
                return False
            
            # Determine candidates for the next element
            # It must be > current_max (since row is increasing)
            # And its diffs with all previous elements must not be in forbidden or used_local
            
            # Optimization: We can limit the search space.
            # The smallest possible next element is current_max + 1.
            # The largest is max_scope.
            
            # To add diversity, we can sometimes shuffle or pick randomly from valid candidates
            # But for finding *any* solution, sequential is often fastest.
            # Let's try sequential first, and if it fails quickly, we might need a different strategy.
            # Actually, sequential DFS can get stuck in deep branches.
            # Let's use a "light" randomization: try candidates in a slightly randomized order?
            # No, for minimizing scope, we want to try small values first.
            
            # However, if we are just trying to fill a row within a scope, 
            # we can use a randomized DFS to escape local traps.
            
            # Let's stick to sequential for speed, but allow backtracking.
            
            # We need to check all possible x from current_max + 1 to max_scope.
            # This can be expensive if max_scope is large.
            # Optimization: Pre-compute valid diffs?
            
            # Let's iterate x
            for x in range(current_max + 1, max_scope + 1):
                # Check diffs
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in forbidden or d in used_local:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check for duplicate diffs within the new set?
                # The set logic above handles 'used_local' and 'forbidden'.
                # We also need to ensure new_diffs don't collide with each other?
                # Since row is strictly increasing, x - a_i are distinct for distinct a_i.
                # So no internal collision in new_diffs.
                
                # Recurse
                row.append(x)
                used_local.update(new_diffs)
                
                if dfs(idx + 1, x):
                    return True
                
                # Backtrack
                row.pop()
                for d in new_diffs:
                    used_local.remove(d)
                    
            return False
            
        # Initial call
        # We start with row=[0], idx=1 (next element to place), current_max=0
        # Note: The 'forbidden' set contains diffs from OTHER rows.
        # 'used_local' contains diffs within the CURRENT row being built.
        
        if dfs(1, 0):
            return list(row)
        return None

    # ---- Local Search with Row Reconstruction ----
    def local_search(start, rng, steps, T0=3.0):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 200 == 0 and remaining() < 10:
                break
            
            r = rng.randrange(n_rows)
            row = cur[r]
            
            # Move Type 1: Point Mutation (Fast)
            # Move Type 2: Row Reconstruction (Slower, but structural)
            
            if rng.random() < 0.7:
                # Point Mutation
                # Pick a random position (prefer high values to reduce scope)
                if rng.random() < 0.5:
                    p = row.index(max(row))
                else:
                    p = rng.randrange(1, row_len)
                
                lo = row[p - 1] + 1
                hi = row[p] - 1
                if lo > hi:
                    continue
                    
                # Try to reduce the value
                # Sample a new value
                v = rng.randrange(lo, hi + 1)
                if v == row[p]:
                    continue
                
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                
                other = cur_used - rowd[r]
                if nd & other:
                    continue
            else:
                # Row Reconstruction
                # Try to rebuild row r with a scope slightly smaller than current max
                # or just any valid row to diversify
                other = cur_used - rowd[r]
                
                # Try to fit within current scope first, then slightly larger
                # If we want to improve, we should try to fit within cur_scope - 1
                target_scope = row_maxes[r] - 1
                if target_scope < row_len: # Must be at least k
                    target_scope = row_len
                    
                new_row = backtrack_row(other, target_scope, rng, timeout=0.2)
                
                if new_row is None:
                    # Try to fit within current scope (diversify)
                    new_row = backtrack_row(other, row_maxes[r], rng, timeout=0.2)
                    
                if new_row is None:
                    continue
                
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                if nd & other:
                    continue

            # Evaluate new state
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)

            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)

            new_sum = cur_sum - row_sums[r] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj

            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other | nd
                row_sums[r] = new_sum_row
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            T = max(0.05, T * 0.999)

        return best

    # ---- Main Search ----
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(42)

    # Phase 1: Randomized Greedy to find a good starting point
    while remaining() > 20:
        bias = rng.uniform(0.5, 0.95)
        window = rng.choice([3, 5, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows

    # Phase 2: Local Search to polish
    # We run local search from the best found so far
    # And also from a few other good candidates
    
    candidates = [best_rows]
    
    # Add a few more diverse candidates
    for _ in range(3):
        if remaining() < 15:
            break
        bias = rng.uniform(0.5, 0.9)
        window = rng.choice([5, 8])
        rows = rand_greedy(rng, bias, window)
        if scope_of(rows) < best_scope + 10:
            candidates.append(rows)

    for cand in candidates:
        if remaining() < 10:
            break
        # Run local search
        # Adjust steps based on remaining time
        est_steps = int(remaining() * 50) 
        est_steps = max(500, min(est_steps, 5000))
        
        res = local_search(cand, rng, est_steps, T0=rng.uniform(2.0, 5.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res

    # Final Check
    if not is_valid(best_rows):
        best_rows = base

    return best_rows
