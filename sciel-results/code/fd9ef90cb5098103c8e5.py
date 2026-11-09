"""
Difference Triangle Set (7,5) - Optimized Combined Strategy
Uses multiple initialization strategies, focused simulated annealing
targeting the maximum-scope row, and efficient final polish.
"""
import time
import random
import sys
import heapq

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 142s to allow for cleanup/return
    deadline = start_time + 142.0

    def elapsed():
        return time.time() - start_time

    def scope_of(rows):
        if not rows:
            return float('inf')
        return max(max(r) for r in rows)

    def get_diffs(row):
        """Calculate all unique positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ai)
        return diffs

    def build_greedy_optimal(n_rows, k, seed=0):
        """
        Greedy construction with lookahead: build rows one by one,
        but when choosing the next value for a row, prefer values that
        minimize the maximum future requirement.
        """
        rnd = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                # Find all valid x values up to some limit
                valid_xs = []
                check_limit = min(x + 20, 150)
                while x <= check_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
                
                if not valid_xs:
                    return None
                
                # Prefer smaller values, but with some randomness
                if len(valid_xs) > 1 and rnd.random() < 0.3:
                    # Pick from top 3 smallest with some randomness
                    top3 = valid_xs[:3]
                    chosen = rnd.choice(top3)
                else:
                    chosen = valid_xs[0]
                
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen + 1
            
            rows.append(row)
        return rows

    def build_compact_rows(n_rows, k, max_scope):
        """
        Build rows with a hard scope limit. Returns None if impossible.
        Uses a more aggressive strategy: always pick the smallest valid value.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if x > max_scope:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    def build_interleaved(n_rows, k, max_scope=None):
        """
        Greedily build rows interleaved: extend the row with the smallest
        current maximum. This keeps rows balanced and compact.
        """
        rows = [[0] for _ in range(n_rows)]
        used = set()
        max_x = max_scope if max_scope is not None else 200

        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            # Pick the row with the smallest current max
            i = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[i]
            x = row[-1] + 1
            found = False
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                return None

        # Verify completion
        for r in rows:
            if len(r) != k + 1:
                return None
        return rows

    def build_sequential(n_rows, k):
        """
        Simple sequential greedy. Builds one row completely before starting the next.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    # --- Phase 1: Initialization ---
    # Try multiple strategies and keep the best
    candidates = []
    
    # Strategy 1: Interleaved greedy
    rows_int = build_interleaved(n_rows, k)
    if rows_int is not None:
        candidates.append(rows_int)
    
    # Strategy 2: Sequential greedy
    rows_seq = build_sequential(n_rows, k)
    if rows_seq is not None:
        candidates.append(rows_seq)
    
    # Strategy 3: Greedy with lookahead (multiple seeds)
    for seed in [0, 1, 2, 42, 123]:
        if elapsed() > 3.0:
            break
        rows_gl = build_greedy_optimal(n_rows, k, seed=seed)
        if rows_gl is not None:
            candidates.append(rows_gl)
    
    # Strategy 4: Compact rows with decreasing scope limits
    for scope_limit in [110, 108, 106, 105]:
        if elapsed() > 5.0:
            break
        rows_compact = build_compact_rows(n_rows, k, scope_limit)
        if rows_compact is not None:
            candidates.append(rows_compact)
            break  # Found one, no need to try smaller limits
    
    if not candidates:
        # Fallback
        return [[0, 1, 2, 3, 4, 5]] * n_rows
    
    # Pick the best candidate
    best_rows = min(candidates, key=scope_of)
    best_scope = scope_of(best_rows)

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 2: Simulated Annealing with Row Replacement ---
    # This is the structural improvement from the inspiration.
    # We replace entire rows to allow large structural changes.
    
    def replace_row_fast(rows, ri, used, k, max_mark, rnd, randomize_prob=0.0):
        """
        Fast row replacement: try to build a new row with max <= max_mark
        that doesn't conflict with used differences.
        """
        old_diffs = get_diffs(rows[ri])
        used_other = used - old_diffs
        
        # Precompute: for each possible x, which differences would it create with existing row elements?
        # We'll just do a direct scan but with early termination
        
        for attempt in range(10):
            row = [0]
            local_diffs = set()
            x = 1
            success = True
            
            while len(row) < k + 1:
                if x > max_mark:
                    success = False
                    break
                
                # Check if x conflicts
                conflict = False
                for a in row:
                    dv = x - a
                    if dv in used_other or dv in local_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept x
                    for a in row:
                        local_diffs.add(x - a)
                    row.append(x)
                
                # Next x: always increment by 1, or skip if randomize
                if randomize_prob > 0 and rnd.random() < randomize_prob and x < max_mark:
                    x += 2  # Skip one value
                else:
                    x += 1
            
            if success and len(row) == k + 1:
                # Final verification
                new_diffs = get_diffs(row)
                if not (new_diffs & used_other):
                    return row, used_other | new_diffs
        
        return None, None

    def simulated_annealing(rows, n_rows, k, time_limit, rnd):
        current = [r[:] for r in rows]
        best = [r[:] for r in rows]
        
        used = set()
        for r in current:
            used |= get_diffs(r)
            
        cur_scope = scope_of(current)
        best_scope = cur_scope
        
        deadline_local = time.time() + time_limit
        T = 30.0
        cooling = 0.9995
        min_T = 0.05
        
        # Track which row has the maximum scope
        def max_scope_row_idx():
            max_val = -1
            max_idx = 0
            for i, r in enumerate(current):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_idx = i
            return max_idx
        
        while time.time() < deadline_local and T > min_T:
            # Focus on the row with the maximum scope (70% of the time)
            if rnd.random() < 0.7:
                ri = max_scope_row_idx()
            else:
                ri = rnd.randint(0, n_rows - 1)
            
            old_row_max = current[ri][-1]
            
            # Try to reduce this row's maximum
            # Target: old_row_max - 1, then old_row_max - 2, etc.
            found_new_row = False
            for reduction in range(1, 4):  # Try reducing by 1, 2, or 3
                target_max = old_row_max - reduction
                if target_max < 5:  # Minimum reasonable scope
                    break
                
                new_row, new_used = replace_row_fast(current, ri, used, k, target_max, rnd)
                
                if new_row is not None:
                    # Calculate new global scope
                    max_other = max(r[-1] for i, r in enumerate(current) if i != ri) if n_rows > 1 else 0
                    actual_new_scope = max(max_other, new_row[-1])
                    delta = actual_new_scope - cur_scope
                    
                    # Accept if improvement or with probability
                    if delta < 0 or (T > 0 and rnd.random() < pow(2.71828, -delta / T)):
                        current[ri] = new_row
                        used = new_used
                        cur_scope = actual_new_scope
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best = [r[:] for r in current]
                        found_new_row = True
                        break
            
            if not found_new_row:
                # Try shuffling a non-max row to free up differences
                if rnd.random() < 0.2:
                    non_max_rows = [i for i in range(n_rows) if i != max_scope_row_idx()]
                    if non_max_rows:
                        ri2 = rnd.choice(non_max_rows)
                        new_row, new_used = replace_row_fast(current, ri2, used, k, current[ri2][-1], rnd, randomize_prob=0.3)
                        if new_row is not None:
                            current[ri2] = new_row
                            used = new_used
                            # Scope doesn't change, but difference set does
                            # This might help future reductions
            
            T *= cooling
        
        return best, best_scope

    # --- Phase 2: Simulated Annealing ---
    if elapsed() < deadline - 8.0:
        rng_sa = random.Random(42)
        time_budget = min(deadline - time.time() - 5.0, 100.0)
        improved_rows, improved_scope = simulated_annealing(best_rows, n_rows, k, time_budget, rng_sa)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        
        # Second SA pass with different seed if time permits and improvement was found
        if improved_scope < best_scope and elapsed() < deadline - 15.0:
            rng_sa2 = random.Random(12345)
            time_budget2 = deadline - time.time() - 5.0
            if time_budget2 > 10.0:
                improved_rows2, improved_scope2 = simulated_annealing(best_rows, n_rows, k, time_budget2, rng_sa2)
                if improved_scope2 < best_scope:
                    best_scope = improved_scope2
                    best_rows = improved_rows2

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 3: Final Deterministic Polish ---
    # Try to reduce individual elements by shifting them down.
    # This cleans up any "wasted" space left by SA.
    
    def final_polish(rows, n_rows, k):
        rows = [r[:] for r in rows]
        used = set()
        for r in rows:
            used |= get_diffs(r)
        
        best_scope = scope_of(rows)
        
        # Multiple passes to ensure convergence
        for _pass in range(3):
            improved = True
            while improved:
                improved = False
                for ri in range(n_rows):
                    row = rows[ri]
                    for pi in range(1, len(row)):
                        old_val = row[pi]
                        # Try to move row[pi] down by 1, 2, 3 (limited search)
                        for delta in range(1, 4):
                            new_val = old_val - delta
                            if new_val <= row[pi-1]:
                                break
                            if pi < len(row) - 1 and new_val >= row[pi+1]:
                                break
                            
                            # Calculate diff changes efficiently
                            old_diffs = set()
                            new_diffs = set()
                            
                            for i in range(pi):
                                old_diffs.add(old_val - row[i])
                                new_diffs.add(new_val - row[i])
                            
                            for j in range(pi + 1, len(row)):
                                old_diffs.add(row[j] - old_val)
                                new_diffs.add(row[j] - new_val)
                            
                            remaining_used = used - old_diffs
                            if not (new_diffs & remaining_used):
                                # Valid move
                                used = remaining_used | new_diffs
                                row[pi] = new_val
                                new_scope = max(max(r) for r in rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                improved = True
                                break
        
        return rows, best_scope

    # --- Phase 3: Final Deterministic Polish ---
    if elapsed() < deadline - 2.0:
        polished_rows, polished_scope = final_polish(best_rows, n_rows, k)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_rows = polished_rows
        
        # One more polish pass if time permits
        if polished_scope < best_scope and elapsed() < deadline - 1.0:
            polished_rows2, polished_scope2 = final_polish(best_rows, n_rows, k)
            if polished_scope2 < best_scope:
                best_scope = polished_scope2
                best_rows = polished_rows2

    return best_rows

if __name__ == "__main__":
    # Quick self-test
    rows = construct(7, 5)
    print(f"Scope: {max(max(r) for r in rows)}")
    print(rows)
