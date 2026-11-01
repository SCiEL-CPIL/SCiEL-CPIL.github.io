"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=200):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Add a bit of randomness to escape local minima in row construction
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 150):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        max_passes = 5
        while improved and time_left() > 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=150)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # ---------- Hybrid ILS + SA ----------
    # We will run multiple cycles. Each cycle:
    # 1. Takes the current best.
    # 2. Applies a kick (perturbation).
    # 3. Runs a short SA to refine.
    # 4. If better, updates best.
    
    # Order rows by descending scope for SA focus
    order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
    
    # Helper for SA
    def setup_sa(rows):
        """Initialize SA state structures."""
        sa_rows = [list(best_rows[i]) for i in order]
        max_diff = max(max(r) for r in sa_rows) + 1
        conflict = [0] * (max_diff + 1)
        for row in sa_rows:
            for d in row_diffs_list(row):
                if d >= len(conflict):
                    conflict.extend([0] * (d + 1 - len(conflict)))
                conflict[d] += 1
        
        cur_conflict_count = sum(1 for c in conflict if c > 1)
        
        state = {
            'rows': sa_rows,
            'conflict': conflict,
            'cur_conf': cur_conflict_count,
            'cur_scope': max(sa_rows[0]),
            'best_rows': [list(r) for r in sa_rows],
            'best_conf': cur_conflict_count,
            'best_scope': max(sa_rows[0]),
            'row_diffs_sets': [set(row_diffs_set(r)) for r in sa_rows]
        }
        return state

    def update_conflict(state, d, delta):
        conflict = state['conflict']
        old = conflict[d]
        new = old + delta
        if old == 1 and new == 2:
            state['cur_conf'] += 1
        elif old == 2 and new == 1:
            state['cur_conf'] -= 1
        conflict[d] = new

    def remove_row_diffs(state, row):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d >= len(state['conflict']):
                    # This shouldn't happen if conflict array is large enough, but safe
                    state['conflict'].extend([0] * (d + 1 - len(state['conflict'])))
                update_conflict(state, d, -1)

    def add_row_diffs(state, row):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d >= len(state['conflict']):
                    state['conflict'].extend([0] * (d + 1 - len(state['conflict'])))
                update_conflict(state, d, 1)

    def greedy_row_for(used_set):
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used_set) and not (new & row_diffs):
                row.append(x)
                row_diffs |= new
        return row

    def random_row_for(used_set, jitter=0.15):
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used_set) and not (new & row_diffs):
                if jitter and random.random() < jitter and x < 400:
                    continue
                row.append(x)
                row_diffs |= new
        return row

    def kick_solution(state):
        """Perturb the current SA solution by replacing 1-2 rows with random ones."""
        rows = state['rows']
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other.update(state['row_diffs_sets'][j])
            
            # Rebuild row with high jitter
            new_row = random_row_for(used_other, jitter=0.4)
            if len(new_row) == k + 1:
                # Update state structures
                old_row = rows[idx]
                remove_row_diffs(state, old_row)
                rows[idx] = new_row
                add_row_diffs(state, new_row)
                state['row_diffs_sets'][idx] = set(row_diffs_set(new_row))
                state['cur_scope'] = max(rows[0]) # Scope is defined by max of row 0 in SA order

    # Main ILS loop
    cycles = 0
    while time_left() > 20:
        cycles += 1
        state = setup_sa(best_rows)
        
        # Apply kick
        if cycles > 1: # Don't kick on first cycle, just SA
            kick_solution(state)
        
        # Short SA phase
        t = 1.0
        t_min = 0.1
        cooling = 0.995
        steps = 0
        max_steps = 2000 # Limit steps per cycle to ensure time management
        
        while time_left() > 15 and t > t_min and steps < max_steps:
            steps += 1
            # Pick a row: bias toward the largest (index 0) early
            if random.random() < 0.5:
                idx = 0
            else:
                idx = random.randrange(n_rows)

            old_row = state['rows'][idx]
            
            # Remove old row
            remove_row_diffs(state, old_row)
            conf_after_remove = state['cur_conf']
            
            # Build used set
            used_set = set()
            for j in range(n_rows):
                if j != idx:
                    used_set.update(state['row_diffs_sets'][j])
            
            # Propose new row
            r = random.random()
            if r < 0.4:
                new_row = greedy_row_for(used_set)
            elif r < 0.8:
                new_row = random_row_for(used_set, jitter=0.15)
            else:
                new_row = random_row_for(used_set, jitter=0.3)
            
            if len(new_row) != k + 1:
                add_row_diffs(state, old_row)
                t *= cooling
                continue
            
            # Add new row
            add_row_diffs(state, new_row)
            conf_after_add = state['cur_conf']
            
            new_scope = max(new_row) if idx == 0 else max(state['rows'][0])
            conf_delta = conf_after_add - conf_after_remove
            scope_delta = new_scope - state['cur_scope']
            
            accept = False
            if conf_delta < 0 or (conf_delta == 0 and scope_delta < 0):
                accept = True
            else:
                penalty = conf_delta * 1000 + scope_delta
                if penalty < 0:
                    accept = True
                else:
                    if random.random() < math.exp(-penalty / max(t, 1e-6)):
                        accept = True
            
            if accept:
                state['rows'][idx] = new_row
                state['row_diffs_sets'][idx] = set(row_diffs_set(new_row))
                state['cur_conf'] = conf_after_add
                state['cur_scope'] = new_scope
                
                cur_energy = (state['cur_conf'], state['cur_scope'])
                best_energy = (state['best_conf'], state['best_scope'])
                if cur_energy < best_energy:
                    state['best_conf'] = state['cur_conf']
                    state['best_scope'] = state['cur_scope']
                    state['best_rows'] = [list(r) for r in state['rows']]
            else:
                # Revert
                remove_row_diffs(state, new_row)
                add_row_diffs(state, old_row)
            
            t *= cooling
        
        # Check if SA found a better valid solution
        if state['best_conf'] == 0:
            # Map back to original order
            restored = [None] * n_rows
            for i in range(n_rows):
                restored[order[i]] = state['best_rows'][i]
            if is_valid(restored):
                s = scope(restored)
                if s < best_scope:
                    best_rows = restored
                    best_scope = s
                    # Apply local polish immediately if we improved
                    if time_left() > 15:
                        polished = local_polish(best_rows)
                        if scope(polished) < best_scope:
                            best_rows = polished
                            best_scope = scope(best_rows)

    # Final polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
