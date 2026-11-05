"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    
    def scope_of(rows):
        return max(max(r) for r in rows) if rows else 0

    def compute_cost(rows):
        """
        Calculate cost = scope + penalty * conflicts.
        Conflicts are differences that appear more than once globally.
        """
        used = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        used[d] += 1
                    else:
                        used[d] = 1
        
        num_conflicts = 0
        for count in used.values():
            if count > 1:
                num_conflicts += count - 1
        
        # Large penalty to drive towards validity
        penalty = 10000
        return max_entry + penalty * num_conflicts

    def is_valid(rows):
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

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def local_search_polish(rows, time_limit):
        """Greedy local search to polish a valid solution."""
        rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])
        
        # Precompute diffs for speed
        def get_row_diffs(row):
            s = set()
            for i in range(L):
                for j in range(i+1, L):
                    s.add(row[j]-row[i])
            return s
            
        row_diffs = [get_row_diffs(r) for r in rows]
        all_diffs = set()
        for s in row_diffs:
            all_diffs |= s
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for ri in range(n):
                for ci in range(1, L):
                    old_val = rows[ri][ci]
                    lower = rows[ri][ci-1] + 1
                    upper = (rows[ri][ci+1] - 1) if ci + 1 < L else best_scope
                    
                    # Try to reduce the value
                    for nv in range(old_val - 1, lower - 1, -1):
                        if nv < 1: break
                        new_row = rows[ri][:]
                        new_row[ci] = nv
                        new_diffs = get_row_diffs(new_row)
                        if len(new_diffs) != diffs_per_row: continue
                        
                        other_diffs = all_diffs - row_diffs[ri]
                        if new_diffs & other_diffs: continue
                        
                        # Valid move
                        new_scope = max(best_scope, max(new_row))
                        if new_scope < best_scope:
                            all_diffs = (all_diffs - row_diffs[ri]) | new_diffs
                            row_diffs[ri] = new_diffs
                            rows[ri] = new_row
                            best_scope = new_scope
                            improved = True
                            break
        return rows, best_scope

    # Main Search
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # Phase 1: Randomized Greedy to get diverse starting points
    # We run a few restarts to see if we can find a better valid start
    attempts = 0
    while time.time() < deadline - 60.0 and attempts < 20:
        attempts += 1
        rows = randomized_greedy(window=50 + random.randint(0, 50))
        if is_valid(rows):
            sc = scope_of(rows)
            if sc < best_scope:
                best_rows = rows
                best_scope = sc
                if best_scope <= 112:
                    break

    # Phase 2: Simulated Annealing
    # We run SA on the best valid solution found so far, or on random valid/invalid states?
    # It's safer to start from a valid solution and use a cost function that penalizes invalidity,
    # allowing it to jump over invalid states to find better valid ones.
    
    # We will run multiple SA runs if time permits
    sa_runs = 0
    max_sa_runs = 10
    while time.time() < deadline - 10.0 and sa_runs < max_sa_runs:
        sa_runs += 1
        # Initialize SA with the best known valid solution
        current_rows = [r[:] for r in best_rows]
        current_cost = compute_cost(current_rows)
        
        T_start = 50.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        it = 0
        max_iters = 50000
        last_improvement = 0
        
        while T > T_end and it < max_iters and time.time() < deadline - 5.0:
            it += 1
            
            # Generate a move
            move_type = random.random()
            r_idx = random.randint(0, n_rows - 1)
            
            new_rows = [row[:] for row in current_rows]
            
            if move_type < 0.7:
                # Single element shift
                c_idx = random.randint(1, k)
                old_val = new_rows[r_idx][c_idx]
                
                # Determine range for new value
                min_v = new_rows[r_idx][c_idx - 1] + 1
                # We can allow going up to current scope + some margin to escape
                max_v = min(old_val + 10, best_scope + 20)
                if c_idx < k:
                    max_v = min(max_v, new_rows[r_idx][c_idx + 1] - 1)
                
                if min_v > max_v:
                    continue
                    
                new_val = random.randint(min_v, max_v)
                if new_val == old_val:
                    continue
                    
                new_rows[r_idx][c_idx] = new_val
                
            else:
                # Suffix shift: add delta to a[c_idx..k]
                c_idx = random.randint(1, k)
                # delta can be negative or positive
                # But we must maintain strict increase
                # a[c_idx] + delta > a[c_idx-1]
                # a[k] + delta < infinity (but we want to keep scope low)
                
                # Estimate max delta
                # a[c_idx] + delta > a[c_idx-1] => delta > a[c_idx-1] - a[c_idx]
                # a[k] + delta < current_max + margin
                min_delta = new_rows[r_idx][c_idx - 1] - new_rows[r_idx][c_idx] + 1
                # Also a[i] + delta < a[i+1] + delta is always true if a[i] < a[i+1]
                # So internal order is preserved.
                # We just need to ensure it doesn't exceed the "current" max too much or go negative.
                # Actually, if c_idx=1, a[0]=0. a[1]+delta > 0 => delta > -a[1].
                
                max_delta = 10 # Small shifts
                min_delta = max(min_delta, -10)
                
                if min_delta > max_delta:
                    continue
                
                delta = random.randint(min_delta, max_delta)
                if delta == 0:
                    continue
                
                for j in range(c_idx, k + 1):
                    new_rows[r_idx][j] += delta
                
                # Check if any value became invalid (e.g., <= 0 or not increasing)
                valid_move = True
                if new_rows[r_idx][0] != 0: valid_move = False # Should not happen
                for j in range(1, k+1):
                    if new_rows[r_idx][j] <= new_rows[r_idx][j-1]:
                        valid_move = False
                        break
                    if new_rows[r_idx][j] < 0:
                        valid_move = False
                        break
                if not valid_move:
                    continue

            new_cost = compute_cost(new_rows)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                current_rows = new_rows
                current_cost = new_cost
                
                # Check if we found a better valid solution
                # If cost < best_scope, it must be valid and have scope < best_scope
                # (Because penalty is 10000, any conflict adds > 10000)
                if current_cost < best_scope:
                    # Verify validity to be absolutely sure
                    if is_valid(current_rows):
                        sc = scope_of(current_rows)
                        if sc < best_scope:
                            best_scope = sc
                            best_rows = [row[:] for row in current_rows]
                            last_improvement = it
                            if best_scope <= 112:
                                break

            T *= cooling_rate
            if time.time() > deadline - 5.0:
                break

    # Phase 3: Polish the best solution found
    if time.time() < deadline - 2.0:
        polished, polished_scope = local_search_polish(best_rows, deadline - 1.0)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    return best_rows
