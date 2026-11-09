import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    def get_max_scope(rows):
        return max(max(r) for r in rows)

    def generate_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs

    rows, used_diffs = generate_greedy()
    current_scope = get_max_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    # Phase 1: Aggressive Hill Climbing
    improved = True
    max_hc_iters = 50
    hc_iter = 0
    while improved and time.time() < deadline - 10:
        improved = False
        hc_iter += 1
        # Shuffle row order for diversity in later iterations
        row_order = list(range(n_rows))
        if hc_iter > 1:
            random.shuffle(row_order)
        
        for r_idx in row_order:
            if time.time() > deadline - 10:
                break
            row = rows[r_idx]
            for pos in range(1, k + 1):
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < k else old_val
                
                # Try to reduce value (most important for scope)
                for new_val in range(low, old_val):
                    old_involved = get_involved_diffs(row, pos, old_val)
                    used_diffs.difference_update(old_involved)
                    
                    new_involved = get_involved_diffs(row, pos, new_val)
                    
                    if len(new_involved) != len(old_involved):
                        used_diffs |= old_involved
                        continue
                    
                    if new_involved & used_diffs:
                        used_diffs |= old_involved
                        continue
                    
                    # Valid
                    used_diffs |= new_involved
                    row[pos] = new_val
                    
                    new_scope = get_max_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        break
                    else:
                        # Revert
                        used_diffs.difference_update(new_involved)
                        used_diffs |= old_involved
                        row[pos] = old_val
                
                if improved:
                    break
            if improved:
                break

    # Phase 2: Improved Simulated Annealing
    T = 5.0
    cooling = 0.998
    min_T = 0.05
    
    sa_step_count = 0
    while time.time() < deadline - 2:
        sa_step_count += 1
        
        # Every 200 steps, do a quick local improvement
        if sa_step_count % 200 == 0:
            improved_hc = True
            while improved_hc and time.time() < deadline - 2:
                improved_hc = False
                for r_idx2 in range(n_rows):
                    row2 = rows[r_idx2]
                    for pos2 in range(1, k + 1):
                        old_val2 = row2[pos2]
                        low2 = row2[pos2 - 1] + 1
                        # Only try to reduce
                        for new_val2 in range(low2, old_val2):
                            old_inv2 = get_involved_diffs(row2, pos2, old_val2)
                            used_diffs.difference_update(old_inv2)
                            new_inv2 = get_involved_diffs(row2, pos2, new_val2)
                            if len(new_inv2) != len(old_inv2):
                                used_diffs |= old_inv2
                                continue
                            if new_inv2 & used_diffs:
                                used_diffs |= old_inv2
                                continue
                            used_diffs |= new_inv2
                            row2[pos2] = new_val2
                            new_scope2 = get_max_scope(rows)
                            if new_scope2 < current_scope:
                                current_scope = new_scope2
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in rows]
                                improved_hc = True
                                break
                            else:
                                used_diffs.difference_update(new_inv2)
                                used_diffs |= old_inv2
                                row2[pos2] = old_val2
                        if improved_hc:
                            break
                    if improved_hc:
                        break
        
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random position
        pos = random.randint(1, k)
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < k else old_val + 30
        
        # Smart move generation
        max_in_row = max(row)
        if old_val == max_in_row:
            # Strong bias to reduce the maximum
            if low < old_val:
                # Try values closer to low (more aggressive reduction)
                if random.random() < 0.6:
                    new_val = random.randint(low, max(low + (old_val - low) // 3, low))
                else:
                    new_val = random.randint(low, old_val - 1)
            else:
                continue
        else:
            # For non-max elements, try both directions
            if random.random() < 0.4:
                # Try to reduce
                search_low = low
                search_high = min(high, old_val - 1)
            else:
                # Try to increase (for repacking)
                search_low = max(low, old_val + 1)
                search_high = min(high, old_val + 20)
            
            if search_low > search_high:
                # Fallback: try nearby
                search_low = max(low, old_val - 10)
                search_high = min(high, old_val + 10)
                if search_low > search_high:
                    continue
            
            new_val = random.randint(search_low, search_high)
        
        if new_val == old_val:
            continue
        
        old_involved = get_involved_diffs(row, pos, old_val)
        used_diffs.difference_update(old_involved)
        
        new_involved = get_involved_diffs(row, pos, new_val)
        
        if len(new_involved) != len(old_involved):
            used_diffs |= old_involved
            continue
        
        if new_involved & used_diffs:
            used_diffs |= old_involved
            continue
        
        # Valid move
        used_diffs |= new_involved
        row[pos] = new_val
        
        new_scope = get_max_scope(rows)
        
        if new_scope < current_scope:
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            delta = new_scope - current_scope
            if random.random() < math.exp(-delta / T):
                current_scope = new_scope
            else:
                # Revert
                used_diffs.difference_update(new_involved)
                used_diffs |= old_involved
                row[pos] = old_val
        
        T *= cooling
        if T < min_T:
            # Restart with new temperature and random perturbation
            T = 5.0
            # Perturb: make a few random valid moves to escape local minimum
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                pr_idx = random.randint(0, n_rows - 1)
                prow = rows[pr_idx]
                ppos = random.randint(1, k)
                pold = prow[ppos]
                plow = prow[ppos - 1] + 1
                phigh = prow[ppos + 1] - 1 if ppos < k else pold + 15
                if plow > phigh:
                    continue
                pnew = random.randint(plow, phigh)
                if pnew == pold:
                    continue
                pold_inv = get_involved_diffs(prow, ppos, pold)
                used_diffs.difference_update(pold_inv)
                pnew_inv = get_involved_diffs(prow, ppos, pnew)
                if len(pnew_inv) != len(pold_inv) or (pnew_inv & used_diffs):
                    used_diffs |= pold_inv
                    continue
                used_diffs |= pnew_inv
                prow[ppos] = pnew
                current_scope = get_max_scope(rows)

    return best_rows
