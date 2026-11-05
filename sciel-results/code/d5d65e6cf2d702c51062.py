import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Greedy construction with multiple strategies
    def greedy_construct(strategy='standard'):
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                elif strategy == 'skip2' and not new & used:
                    row.append(x)
                    used |= new
                x += 1
            rows.append(row)
        return rows
    
    def greedy_construct_packing():
        """Greedy that tries to pack differences tightly."""
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            rows.append(row)
        return rows
    
    # Simulated annealing for optimization
    def simulated_annealing(initial_rows, max_time=60):
        current = [r[:] for r in initial_rows]
        current_scope = get_scope(current)
        best = [r[:] for r in initial_rows]
        best_scope = current_scope
        
        # Track all differences for quick validation
        all_diffs = set()
        for row in current:
            all_diffs |= row_diffs(row)
        
        def compute_scope(rows):
            return max(max(r) for r in rows)
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        sa_start = time.time()
        no_improve = 0
        max_no_improve = 5000
        
        while time_left() > 5:
            if no_improve > max_no_improve:
                break
            
            # Pick a random row and a random position to modify
            r_idx = random.randint(0, n_rows - 1)
            m_idx = random.randint(1, len(current[r_idx]) - 1)
            
            # Generate a new value
            prev = current[r_idx][m_idx - 1] if m_idx > 0 else 0
            next_val = current[r_idx][m_idx + 1] if m_idx < len(current[r_idx]) - 1 else current[r_idx][m_idx] + 20
            
            # Choose a delta
            delta = random.randint(-8, 8)
            new_val = current[r_idx][m_idx] + delta
            
            # Check bounds
            if new_val <= prev or new_val >= next_val:
                # Try to move the whole suffix
                if m_idx < len(current[r_idx]) - 1:
                    new_val = random.randint(prev + 1, max(prev + 1, next_val - 1))
                    if new_val <= prev or new_val >= next_val:
                        continue
                else:
                    new_val = random.randint(prev + 1, current[r_idx][-1] + 10)
                    if new_val <= prev:
                        continue
            
            old_val = current[r_idx][m_idx]
            current[r_idx][m_idx] = new_val
            
            # Check validity
            new_diffs = row_diffs(current[r_idx])
            valid = True
            for d in new_diffs:
                if d in all_diffs and d not in row_diffs(current[r_idx]):
                    valid = False
                    break
            if not valid:
                # Check against other rows
                for d in new_diffs:
                    if d in all_diffs:
                        # This diff is used by another row
                        valid = False
                        break
            
            # More careful check
            valid = True
            used_diffs = set()
            for i, row in enumerate(current):
                rd = row_diffs(row)
                if i == r_idx:
                    rd = new_diffs
                if rd & used_diffs:
                    valid = False
                    break
                used_diffs |= rd
            
            new_scope = compute_scope(current)
            
            if valid:
                # Update all_diffs
                old_row_diffs = row_diffs([0] + current[r_idx][:m_idx] + [old_val] + current[r_idx][m_idx+1:])
                # Recompute properly
                temp_row = current[r_idx][:]
                temp_row[m_idx] = old_val
                old_rd = row_diffs(temp_row)
                
                all_diffs -= old_rd
                all_diffs |= new_diffs
                
                delta_scope = new_scope - current_scope
                
                if new_scope < best_scope:
                    best = [r[:] for r in current]
                    best_scope = new_scope
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Accept if improvement or with probability
                if delta_scope <= 0:
                    current_scope = new_scope
                    # Keep the change
                else:
                    # Metropolis criterion
                    import math
                    prob = math.exp(-delta_scope / max(T, 0.01))
                    if random.random() < prob:
                        current_scope = new_scope
                    else:
                        # Revert
                        current[r_idx][m_idx] = old_val
                        all_diffs -= new_diffs
                        all_diffs |= old_rd
            else:
                # Revert
                current[r_idx][m_idx] = old_val
                no_improve += 1
            
            T *= cooling_rate
        
        return best, best_scope
    
    def local_search_polish(rows, max_time=10):
        """Try to reduce scope by adjusting individual elements."""
        current = [r[:] for r in rows]
        current_scope = get_scope(current)
        best = [r[:] for r in rows]
        best_scope = current_scope
        
        ls_start = time.time()
        
        while time.time() - ls_start < max_time and time_left() > 5:
            improved = False
            for r_idx in range(n_rows):
                for m_idx in range(1, len(current[r_idx])):
                    orig = current[r_idx][m_idx]
                    for delta in range(-3, 4):
                        if delta == 0:
                            continue
                        new_val = orig + delta
                        if new_val <= 0:
                            continue
                        if m_idx > 1 and new_val <= current[r_idx][m_idx - 1]:
                            continue
                        if m_idx < len(current[r_idx]) - 1 and new_val >= current[r_idx][m_idx + 1]:
                            continue
                        
                        current[r_idx][m_idx] = new_val
                        if is_valid(current):
                            new_scope = get_scope(current)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best = [r[:] for r in current]
                                improved = True
                            else:
                                current[r_idx][m_idx] = orig
                        else:
                            current[r_idx][m_idx] = orig
            if not improved:
                break
        
        return best, best_scope
    
    # Generate initial solutions using different strategies
    best_overall = None
    best_scope_overall = float('inf')
    
    # Strategy 1: Standard greedy
    rows1 = greedy_construct('standard')
    scope1 = get_scope(rows1)
    if scope1 < best_scope_overall:
        best_overall = rows1
        best_scope_overall = scope1
    
    # Run simulated annealing on the best initial
    if time_left() > 20:
        sa_result, sa_scope = simulated_annealing(best_overall, max_time=min(60, time_left() - 15))
        if sa_scope < best_scope_overall:
            best_overall = sa_result
            best_scope_overall = sa_scope
    
    # Local search polish
    if time_left() > 10:
        ls_result, ls_scope = local_search_polish(best_overall, max_time=min(10, time_left() - 5))
        if ls_scope < best_scope_overall:
            best_overall = ls_result
            best_scope_overall = ls_scope
    
    # Try a second SA round with different random seed
    if time_left() > 20:
        random.seed(int(time.time() * 1000) % 100000)
        rows2 = greedy_construct('standard')
        sa_result2, sa_scope2 = simulated_annealing(rows2, max_time=min(40, time_left() - 10))
        if sa_scope2 < best_scope_overall:
            best_overall = sa_result2
            best_scope_overall = sa_scope2
        
        if time_left() > 5:
            ls_result2, ls_scope2 = local_search_polish(best_overall, max_time=min(5, time_left() - 2))
            if ls_scope2 < best_scope_overall:
                best_overall = ls_result2
                best_scope_overall = ls_scope2
    
    return best_overall
