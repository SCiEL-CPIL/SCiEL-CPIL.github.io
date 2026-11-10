import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def compute_diffs(rows):
        """Return set of all differences in rows, or None if invalid."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return None
                    used.add(d)
        return used
    
    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
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
        return rows
    
    def greedy_random_construct(temperature=1.0):
        """Randomized greedy: pick from top candidates with temperature control."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                max_x = current_max + max(30, k * 3)
                while x <= max_x and len(candidates) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: extend search
                    while x <= current_max + 200:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                    if not candidates:
                        return None
                
                # Weighted selection
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Smaller candidates get higher weight
                    min_c = candidates[0]
                    weights = [1.0 / (c - min_c + 1) ** temperature for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            rows.append(row)
        
        # Verify
        if is_valid(rows):
            return rows
        return None
    
    def mrv_backtrack_search(seed, max_steps=5000):
        """MRV-based backtracking: pick the row with smallest available x."""
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        steps_done = 0
        total_steps = n_rows * k
        
        while steps_done < total_steps:
            if time.time() > deadline - 5:
                return None
            
            # Find candidates for each incomplete row
            candidates = []
            
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                
                current_row = rows[r]
                last_val = current_row[-1]
                
                # Find the smallest valid x
                x = last_val + 1
                valid_xs = []
                checks = 0
                while len(valid_xs) < 3 and checks < 200:
                    # Quick check: x itself must not be in used_diffs (diff from 0)
                    if x in used_diffs:
                        x += 1
                        checks += 1
                        continue
                    
                    conflict = False
                    for a in current_row[1:]:
                        if (x - a) in used_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        valid_xs.append(x)
                    
                    x += 1
                    checks += 1
                
                if not valid_xs:
                    return None
                
                min_x = valid_xs[0]
                candidates.append((min_x, r, valid_xs))
            
            if not candidates:
                break
            
            # Sort by min_x (most constrained first)
            candidates.sort(key=lambda t: t[0])
            
            # Pick with some randomness: 70% pick the most constrained, 30% pick randomly from top 3
            if random.random() < 0.7 or len(candidates) == 1:
                min_x, r, valid_xs = candidates[0]
            else:
                min_x, r, valid_xs = random.choice(candidates[:min(3, len(candidates))])
            
            # Pick a value from valid_xs (bias toward smaller)
            if len(valid_xs) == 1:
                next_x = valid_xs[0]
            else:
                # 80% pick smallest, 20% pick randomly
                if random.random() < 0.8:
                    next_x = valid_xs[0]
                else:
                    next_x = random.choice(valid_xs[:min(3, len(valid_xs))])
            
            # Place it
            rows[r].append(next_x)
            for a in rows[r][:-1]:
                used_diffs.add(next_x - a)
            
            steps_done += 1
        
        if steps_done < total_steps:
            return None
        
        if is_valid(rows):
            return rows
        return None
    
    def local_search(rows, time_limit):
        """Simulated annealing with incremental conflict tracking."""
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_diffs = compute_diffs(current_rows)
        
        if current_diffs is None:
            return None, current_scope
        
        best_local_rows = [r[:] for r in current_rows]
        best_local_scope = current_scope
        
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.998
        steps = 0
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick a random row and position to modify
            row_idx = random.randint(0, len(current_rows) - 1)
            row = current_rows[row_idx]
            
            if len(row) <= 1:
                continue
            
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try a small perturbation
            delta = random.randint(-5, 5)
            if delta == 0:
                continue
            new_val = old_val + delta
            
            # Check ordering
            if new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute new differences for this row
            old_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_row_diffs.add(row[j] - row[i])
            
            # Tentatively update
            row[pos] = new_val
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            
            # Check conflicts: new diffs not in (current_diffs - old_row_diffs)
            other_diffs = current_diffs - old_row_diffs
            if new_row_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            # Valid move
            new_diffs = other_diffs | new_row_diffs
            new_scope = max(max(r) for r in current_rows)
            
            if new_scope < best_local_scope:
                best_local_scope = new_scope
                best_local_rows = [r[:] for r in current_rows]
            
            # Accept or reject
            if new_scope <= current_scope:
                current_scope = new_scope
                current_diffs = new_diffs
            else:
                prob = math.exp(-(new_scope - current_scope) / max(temperature, 0.1))
                if random.random() < prob:
                    current_scope = new_scope
                    current_diffs = new_diffs
                else:
                    row[pos] = old_val
            
            temperature *= cooling_rate
            steps += 1
        
        return best_local_rows, best_local_scope
    
    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Multiple construction strategies
    # 2a: Randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline - 30:
        rows = greedy_random_construct(temperature=random.uniform(0.5, 2.0))
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        num_restarts += 1
    
    # 2b: MRV backtracking
    while time.time() < deadline - 20:
        seed = random.randint(0, 10**9)
        rows = mrv_backtrack_search(seed)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
    
    # Phase 3: Local search on best solution
    if best_rows is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        # Split time among several local search runs
        num_ls_runs = 3
        time_per_run = remaining_time / num_ls_runs
        
        for i in range(num_ls_runs):
            if time.time() >= deadline - 5:
                break
            
            # Start from best or a fresh solution
            if i == 0:
                start_rows = best_rows
            else:
                fresh = greedy_random_construct(temperature=1.0)
                if fresh is not None and get_scope(fresh) < best_scope:
                    start_rows = fresh
                else:
                    start_rows = best_rows
            
            ls_rows, ls_scope = local_search(start_rows, time_per_run)
            if ls_rows is not None and ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = [r[:] for r in ls_rows]
                if best_scope <= lower_bound:
                    return best_rows
    
    return best_rows
