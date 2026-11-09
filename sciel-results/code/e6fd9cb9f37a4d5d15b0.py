import time
import random
import math
import sys
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute total number of differences needed
    total_diffs = n_rows * (k + 1) * k // 2
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def greedy_row(used_diffs, max_val, rng, skip_prob=0.0):
        """
        Build a single row of length k+1 starting at 0, with all differences
        not in used_diffs, and max value <= max_val.
        """
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x <= max_val:
                # Optional random skip for diversity
                if rng.random() < skip_prob and x < max_val:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        return row
    
    def greedy_construct(M_limit, seed_val=0, skip_prob=0.0):
        """
        Greedy construction: build rows one by one.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = greedy_row(used_diffs, M_limit, rng, skip_prob)
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        return rows
    
    def interleave_greedy(M_limit, seed_val=0, skip_prob=0.0):
        """
        Interleaved greedy: extend the row with the smallest current maximum at each step.
        Maintains local diffs incrementally for speed.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        local_diffs = [set() for _ in range(n_rows)]
        
        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            
            target_row_idx = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target_row_idx]
            ldiffs = local_diffs[target_row_idx]
            
            x = row[-1] + 1
            found = False
            while x <= M_limit:
                if rng.random() < skip_prob and x < M_limit:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in ldiffs:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    ldiffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        
        for r in rows:
            if len(r) != k + 1:
                return None
        
        return rows
    
    def simulated_annealing(rows, time_budget, seed_val=0):
        """
        Simulated annealing to reduce the scope.
        Moves: replace one row with a new valid row that fits in the current scope.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used = get_used_diffs(current)
        deadline = time.time() + time_budget
        T = 20.0
        cooling = 0.998
        step = 0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and T > 0.05:
            step += 1
            
            # Pick a random row to replace
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            used_without = used - old_diffs
            
            # Try to build a new row within current_scope
            max_mark = current_scope
            new_row = None
            
            # Try more randomizations of greedy with varying skip probs
            for attempt in range(40):
                sp = rng.uniform(0.0, 0.4)
                new_row = greedy_row(used_without, max_mark, rng, skip_prob=sp)
                if new_row is not None:
                    break
            
            if new_row is None:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if new_diffs & used_without:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue
            
            # Compute new scope
            other_max = 0
            for i, r in enumerate(current):
                if i != ri:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))
            
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                T = max(T, 15.0)
                no_improve = 0
            
            T *= cooling
        
        return best, best_scope
    
    def local_search(rows, time_budget, seed_val=0):
        """
        Local search: try to reduce the scope by adjusting individual elements.
        Focus on reducing the maximum element.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used = get_used_diffs(current)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            # Find rows that have the current scope as their max
            scope_rows = [i for i in range(n_rows) if max(current[i]) == current_scope]
            
            if not scope_rows:
                current_scope = max(max(r) for r in current)
                scope_rows = [i for i in range(n_rows) if max(current[i]) == current_scope]
            
            if not scope_rows:
                break
            
            # Pick a row with the max element
            ri = rng.choice(scope_rows)
            row = current[ri]
            
            # Find the position of the max element
            pi = row.index(current_scope)
            
            old_val = row[pi]
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            
            used_without = used - old_diffs
            
            # Try to reduce this value
            new_val = old_val - 1
            found = False
            while new_val > row[pi - 1] and new_val >= 0:
                bad = False
                new_diffs = set()
                for i in range(len(row)):
                    if i == pi:
                        continue
                    if i < pi:
                        d = new_val - row[i]
                        if d <= 0 or d in used_without or d in old_diffs:
                            bad = True
                            break
                        new_diffs.add(d)
                    else:
                        d = row[i] - new_val
                        if d <= 0 or d in used_without or d in old_diffs:
                            bad = True
                            break
                        new_diffs.add(d)
                
                if not bad and not (new_diffs & used_without):
                    current[ri][pi] = new_val
                    used = used_without | new_diffs
                    current_scope = max(max(r) for r in current)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [r[:] for r in current]
                    found = True
                    break
                new_val -= 1
            
            if not found:
                # Try a small random perturbation of a non-max element
                ri2 = rng.randint(0, n_rows - 1)
                pi2 = rng.randint(1, k)
                row2 = current[ri2]
                old_val2 = row2[pi2]
                
                old_diffs2 = set()
                for i in range(len(row2)):
                    for j in range(i + 1, len(row2)):
                        old_diffs2.add(row2[j] - row2[i])
                
                used_without2 = used - old_diffs2
                
                # Try decreasing
                delta = rng.randint(1, 3)
                new_val2 = old_val2 - delta
                if new_val2 > row2[pi2 - 1]:
                    bad = False
                    new_diffs2 = set()
                    for i in range(len(row2)):
                        if i == pi2:
                            continue
                        if i < pi2:
                            d = new_val2 - row2[i]
                            if d <= 0 or d in used_without2:
                                bad = True
                                break
                            new_diffs2.add(d)
                        else:
                            d = row2[i] - new_val2
                            if d <= 0 or d in used_without2:
                                bad = True
                                break
                            new_diffs2.add(d)
                    
                    if not bad and not (new_diffs2 & used_without2):
                        current[ri2][pi2] = new_val2
                        used = used_without2 | new_diffs2
                        current_scope = max(max(r) for r in current)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best = [r[:] for r in current]
        
        return best, best_scope
    
    # Main logic
    best_rows = None
    best_M = float('inf')
    
    # Phase 1: Get a baseline with greedy - try multiple M values and strategies
    start_M = total_diffs
    end_M = total_diffs + 80
    
    for M in range(start_M, end_M):
        if time.time() > deadline - 60:
            break
        
        found_this_M = False
        for seed in range(12):
            if time.time() > deadline - 55:
                break
            
            if seed % 4 == 0:
                rows = greedy_construct(M, seed, skip_prob=0.0)
            elif seed % 4 == 1:
                rows = interleave_greedy(M, seed, skip_prob=0.0)
            elif seed % 4 == 2:
                rows = greedy_construct(M, seed, skip_prob=0.15)
            else:
                rows = interleave_greedy(M, seed, skip_prob=0.1)
            
            if rows is not None and is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                found_this_M = True
                break
        
        if found_this_M and best_M <= M:
            break
    
    # If no baseline found, try larger M
    if best_rows is None:
        for M in range(end_M, end_M + 200):
            if time.time() > deadline - 15:
                break
            rows = greedy_construct(M, 0, skip_prob=0.0)
            if rows is not None and is_valid(rows):
                best_M = max(max(r) for r in rows)
                best_rows = [r[:] for r in rows]
                break
    
    # Emergency fallback
    if best_rows is None:
        M_huge = 3000
        rows = greedy_construct(M_huge, 0, skip_prob=0.0)
        if rows is not None:
            best_rows = rows
            best_M = max(max(r) for r in rows)
        else:
            # Construct a valid solution with large gaps
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 1000
                    row.append(base)
                best_rows.append(row)
            best_M = max(max(r) for r in best_rows)
    
    # Phase 2: Multi-start SA with perturbation
    time_left = deadline - time.time()
    if time_left > 15:
        # First SA run
        sa_budget = min(time_left - 15, 45)
        if sa_budget > 0:
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=42)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows
        
        # Second SA run with different seed
        time_left = deadline - time.time()
        if time_left > 15:
            sa_budget2 = min(time_left - 15, 35)
            if sa_budget2 > 0:
                improved_rows2, improved_M2 = simulated_annealing(best_rows, sa_budget2, seed_val=123)
                if improved_rows2 is not None and improved_M2 < best_M:
                    best_M = improved_M2
                    best_rows = improved_rows2
        
        # Third SA run - perturb best and re-optimize
        time_left = deadline - time.time()
        if time_left > 20:
            rng_perturb = random.Random(999)
            perturbed = [r[:] for r in best_rows]
            # Perturb one random element slightly
            ri = rng_perturb.randint(0, n_rows - 1)
            pi = rng_perturb.randint(1, k)
            old_diffs = set()
            for i in range(len(perturbed[ri])):
                for j in range(i + 1, len(perturbed[ri])):
                    old_diffs.add(perturbed[ri][j] - perturbed[ri][i])
            used_other = get_used_diffs(perturbed) - old_diffs
            
            # Try to replace the perturbed row
            new_row = greedy_row(used_other, best_M, rng_perturb, skip_prob=0.3)
            if new_row is not None:
                perturbed[ri] = new_row
                if is_valid(perturbed):
                    sa_budget3 = min(time_left - 20, 25)
                    if sa_budget3 > 0:
                        improved_rows3, improved_M3 = simulated_annealing(perturbed, sa_budget3, seed_val=777)
                        if improved_rows3 is not None and improved_M3 < best_M:
                            best_M = improved_M3
                            best_rows = improved_rows3
    
    # Phase 3: Local search for final refinement
    time_left = deadline - time.time()
    if time_left > 8:
        ls_budget = min(time_left - 8, 20)
        if ls_budget > 0:
            improved_rows4, improved_M4 = local_search(best_rows, ls_budget, seed_val=777)
            if improved_rows4 is not None and improved_M4 < best_M:
                best_M = improved_M4
                best_rows = improved_rows4
    
    # Final validation
    if best_rows is not None:
        if not is_valid(best_rows):
            # Fallback to a known-valid construction
            M_huge = 5000
            rows = greedy_construct(M_huge, 0, skip_prob=0.0)
            if rows is not None:
                best_rows = rows
            else:
                best_rows = []
                base = 0
                for i in range(n_rows):
                    row = [0]
                    for j in range(1, k + 1):
                        base += 2000
                        row.append(base)
                    best_rows.append(row)
    
    return best_rows

# Test / Main guard
if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", max(max(r) for r in rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
