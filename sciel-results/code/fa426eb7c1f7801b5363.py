import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s

    def time_left():
        return deadline - time.time()

    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            d = row_diffs(row)
            if d & seen:
                return False
            seen |= d
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy construction ----------
    def greedy(seed_val=0):
        random.seed(seed_val) if seed_val > 0 else None
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 200
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + window, x + window + 1000):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    row.append(x + 1)
                x = row[-1]
            rows.append(row)
        return rows

    # ---------- Local polish ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=200):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            jitter = 0.1 + 0.1 * (attempt / max_attempts)
            window = 80 + 40 * (attempt / max_attempts)
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, int(x + window)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if random.random() < jitter and trial_x < 400:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(int(x + window), x + window + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                best_row = list(row)
                best_max = max(row)
        return best_row, best_max

    def local_polish(rows, max_passes=8):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 3:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=150)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    # ---------- Kick ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs(rows[j])

            best_row = None
            best_max = float('inf')
            for attempt in range(30):
                if time_left() < 3:
                    break
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < 0.4 and trial_x < 400:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        for trial_x in range(x + 150, x + 600):
                            new = {trial_x - a for a in row}
                            if not (new & used_other) and not (new & current_row_diffs):
                                row.append(trial_x)
                                current_row_diffs |= new
                                found = True
                                break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]

                if valid_row and max(row) < best_max:
                    best_row = list(row)
                    best_max = max(row)

            if best_row is not None:
                rows[idx] = best_row
            else:
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Beam search for target scope (full solution tracking) ----------
    def beam_search_full(target_scope, beam_width=50, max_time=20):
        start = time.time()
        
        # State: (list_of_rows_completed, curr_row, curr_diffs, used_global)
        # Start with empty rows, beginning row 0
        beam = [([], [0], set(), set())]
        
        while beam and (time.time() - start) < max_time:
            new_beam = []
            for completed_rows, curr_row, curr_diffs, used_global in beam:
                if len(curr_row) == k + 1:
                    # Row complete
                    new_completed = completed_rows + [curr_row]
                    if len(new_completed) == n_rows:
                        # Full solution found
                        return new_completed
                    else:
                        new_beam.append((new_completed, [0], set(), used_global | curr_diffs))
                else:
                    last_val = curr_row[-1]
                    for trial_x in range(last_val + 1, min(target_scope + 1, last_val + 120)):
                        new_diffs = {trial_x - a for a in curr_row}
                        if not (new_diffs & used_global) and not (new_diffs & curr_diffs):
                            new_row = curr_row + [trial_x]
                            new_beam.append((completed_rows, new_row, curr_diffs | new_diffs, used_global))
            
            if not new_beam:
                break
            
            # Prune beam
            new_beam.sort(key=lambda state: (max(state[1]), len(state[3])))
            beam = new_beam[:beam_width]
        
        return None

    # ---------- Main search ----------
    best_rows = greedy(0)
    best_scope = scope(best_rows)
    
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
    
    for i in range(25):
        if time_left() < 20:
            break
        new_rows = greedy(i + 1)
        new_scope = scope(new_rows)
        if new_scope < best_scope:
            best_rows = new_rows
            best_scope = new_scope
            if time_left() > 15:
                polished = local_polish(best_rows)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)
    
    while time_left() > 25:
        if time_left() > 10:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)
        
        if time_left() < 15:
            break
        kicked = kick(best_rows)
        
        if time_left() > 10:
            polished_kicked = local_polish(kicked)
            if scope(polished_kicked) < best_scope:
                best_rows = polished_kicked
                best_scope = scope(polished_kicked)
        
        if time_left() > 30 and random.random() < 0.3:
            aggressive = kick(kick(best_rows))
            if time_left() > 15:
                polished_aggr = local_polish(aggressive)
                if scope(polished_aggr) < best_scope:
                    best_rows = polished_aggr
                    best_scope = scope(polished_aggr)
    
    # Try beam search for best_scope - 1
    if time_left() > 15:
        target = best_scope - 1
        bt_result = beam_search_full(target, beam_width=80, max_time=12)
        if bt_result is not None and len(bt_result) == n_rows:
            if is_valid(bt_result):
                best_rows = bt_result
                best_scope = target
    
    # Final polish
    if time_left() > 5:
        best_rows = [list(r) for r in best_rows]
        for _ in range(3):
            if time_left() < 2:
                break
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
            for i in order:
                if time_left() < 1:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(best_rows[j])
                new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=300)
                if new_max < max(best_rows[i]):
                    best_rows[i] = new_row
                    improved = True
            if not improved:
                break
    
    if not is_valid(best_rows):
        best_rows = greedy(0)
    
    return best_rows
