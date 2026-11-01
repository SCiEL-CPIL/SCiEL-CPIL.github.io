import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound for scope (counting argument)
    # We need total_diffs distinct differences, each row has max diff = scope
    # So scope must be at least total_diffs / n_rows ... but more precisely:
    # Each row with max element m has differences in [1, m], so at most m distinct diffs per row
    # Total distinct diffs needed = total_diffs
    # So sum of max elements >= total_diffs, meaning avg max >= total_diffs / n_rows
    # But we want min of max elements, so at least total_diffs / n_rows
    # Actually the counting bound: if all rows have max <= M, then each row has at most M distinct diffs
    # so total_diffs <= n_rows * M, hence M >= ceil(total_diffs / n_rows)
    counting_bound = (total_diffs + n_rows - 1) // n_rows
    
    # M_max: maximum difference value we'll track
    # Be generous to allow exploration
    M_max = max(500, counting_bound * 3)
    
    # --- Greedy Initialization ---
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # --- Build initial solution ---
    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)
    
    # --- State management for local search ---
    # diff_counts[d] = number of times difference d appears across all rows
    diff_counts = [0] * (M_max + 1)
    
    def rebuild_counts(rows):
        for d in range(len(diff_counts)):
            diff_counts[d] = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= M_max:
                        diff_counts[d] += 1
    
    def num_duplicates():
        return sum(1 for c in diff_counts if c > 1)
    
    def scope_of(rows):
        return max(max(r) for r in rows)
    
    # Initialize
    rebuild_counts(best_rows)
    cur_rows = [r[:] for r in best_rows]
    cur_num_dup = num_duplicates()
    
    # --- Move evaluation and application ---
    def delta_num_dup(ri, ei, new_val):
        """Calculate change in number of duplicate difference values."""
        row = cur_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return 0
        
        delta = 0
        # Removing old differences involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 2:
                    delta -= 1  # 2 -> 1, removes one duplicate value
        
        # Adding new differences involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                c = diff_counts[d]
                if c == 1:
                    delta += 1  # 1 -> 2, creates one duplicate value
        
        return delta
    
    def apply_move(ri, ei, new_val):
        """Apply a move: set cur_rows[ri][ei] = new_val, update diff_counts and cur_num_dup."""
        nonlocal cur_num_dup
        row = cur_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return
        
        # Remove old diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                diff_counts[d] -= 1
        
        cur_rows[ri][ei] = new_val
        
        # Add new diffs
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                diff_counts[d] += 1
        
        # Recompute num_dup efficiently is hard incrementally; recompute
        cur_num_dup = num_duplicates()
    
    # --- Local Search with Simulated Annealing ---
    def local_search(max_steps, T_start=50.0, T_end=0.5, alpha=0.995):
        nonlocal cur_num_dup, best_scope, best_rows
        
        steps = 0
        T = T_start
        no_improve = 0
        max_no_improve = 500
        
        while steps < max_steps and time.time() < deadline:
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            
            row = cur_rows[ri]
            old_val = row[ei]
            
            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else best_scope + 50
            
            if lower > upper:
                steps += 1
                continue
            
            # Sample new value with bias toward smaller values
            # Use a distribution that favors smaller values
            if random.random() < 0.7:
                # Biased toward lower end
                frac = random.random() ** 2
            else:
                frac = random.random()
            new_val = lower + int(frac * (upper - lower + 1))
            
            if new_val == old_val:
                steps += 1
                continue
            
            d_dup = delta_num_dup(ri, ei, new_val)
            
            # Acceptance criteria
            if cur_num_dup > 0:
                # Phase 1: try to eliminate duplicates
                if d_dup < 0:
                    accept = True
                elif d_dup == 0:
                    # Neutral: accept with some probability, bias toward smaller values
                    if new_val < old_val:
                        accept = True
                    else:
                        accept = random.random() < 0.1
                else:
                    # d_dup > 0: increases duplicates, only accept with SA probability
                    accept = random.random() < math.exp(-d_dup / max(T, 0.01))
            else:
                # Phase 2: no duplicates, minimize scope
                if new_val < old_val:
                    accept = True
                elif new_val == old_val:
                    accept = True
                else:
                    # Try to escape local minimum in scope
                    d_scope = new_val - old_val
                    accept = random.random() < math.exp(-d_scope / max(T, 0.01))
            
            if accept:
                apply_move(ri, ei, new_val)
                
                if cur_num_dup == 0:
                    s = scope_of(cur_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [r[:] for r in cur_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T = max(T_end, T * alpha)
            steps += 1
            
            if no_improve > max_no_improve:
                break
        
        return steps > 0
    
    # --- Perturbation / Kick ---
    def kick(strength=3):
        nonlocal cur_num_dup
        # Pick 1-2 rows and move 'strength' elements randomly
        for _ in range(random.randint(1, 2)):
            ri = random.randint(0, n_rows - 1)
            row = cur_rows[ri]
            for _ in range(strength):
                ei = random.randint(1, k)
                old_val = row[ei]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else best_scope + 30
                
                if lower > upper:
                    continue
                
                # Pick a value that's likely different
                if upper - lower >= 5:
                    new_val = random.choice([lower, upper, random.randint(lower, upper)])
                else:
                    new_val = random.randint(lower, upper)
                
                if new_val != old_val:
                    apply_move(ri, ei, new_val)
    
    # --- Main ILS loop ---
    local_search(max_steps=20000, T_start=30.0, T_end=1.0)
    
    iteration = 0
    while time.time() < deadline - 5:
        # If we've hit the counting bound, we're done
        if best_scope <= counting_bound:
            break
        
        # Perturb
        # Stronger kick if stuck (high num_dupes)
        if cur_num_dup > 10:
            kick(strength=4)
        else:
            kick(strength=2)
        
        # Local search to repair and optimize
        # Use higher temperature if many duplicates
        if cur_num_dup > 5:
            local_search(max_steps=30000, T_start=80.0, T_end=2.0)
        else:
            local_search(max_steps=30000, T_start=20.0, T_end=0.5)
        
        iteration += 1
    
    # Final cleanup: try a final greedy-based repair if still invalid
    if num_duplicates() > 0:
        # Try to fix by rebuilding worst rows greedily
        pass
    
    # Ensure we return the best valid solution found
    # If best_rows is valid, return it
    # Verify best_rows
    all_diffs = set()
    valid = True
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    valid = False
                    break
                all_diffs.add(d)
        if not valid:
            break
    
    if valid:
        return best_rows
    
    # Fallback: return current state if it's valid, else reconstruct
    # Rebuild a valid solution from scratch as last resort
    rows = greedy_init()
    return rows
