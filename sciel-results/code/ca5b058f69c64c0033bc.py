import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    def greedy_init(noise=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                candidates = []
                while len(candidates) < 5: # Look ahead a bit if noise
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                    if len(candidates) >= 1 and random.random() > noise:
                         break
                # If noise is 0, we just take the first one.
                # If noise > 0, we might pick a later one.
                # For simplicity, let's just do standard greedy for init.
                if not candidates:
                    # Should not happen
                    continue
                chosen = candidates[0]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_init()
    best_rows = copy.deepcopy(rows)
    best_scope = max(r[-1] for r in best_rows)
    last_improve_time = start_time
    
    T = 100.0
    cooling_rate = 0.9995
    
    def get_scope(rs):
        return max(r[-1] for r in rs)

    def apply_compress_move(r_idx, j_idx):
        """
        Re-greedy the suffix of row r_idx starting at j_idx.
        Modifies rows and used_diffs in place.
        """
        # Remove all diffs of row r_idx
        old_row = rows[r_idx]
        for i in range(len(old_row)):
            for l in range(i + 1, len(old_row)):
                used_diffs.discard(old_row[l] - old_row[i])
                
        # Re-greedy from j_idx
        # Prefix is rows[r_idx][:j_idx]
        new_row = rows[r_idx][:j_idx]
        
        # The next element must be > last of prefix
        # We want to minimize the final scope, so we pick the smallest possible values greedily.
        
        for _ in range(k + 1 - len(new_row)):
            if len(new_row) == 0:
                x = 0
            else:
                x = new_row[-1] + 1
            
            # Find smallest valid x
            while True:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & used_diffs:
                    break
                x += 1
            
            new_row.append(x)
            used_diffs |= new_diffs
            
        rows[r_idx] = new_row

    while time.time() < deadline:
        # Check for stagnation
        if time.time() - last_improve_time > 10:
            # Restart with noise? Or just reset T?
            T = 50.0
            # Maybe re-initialize with a slightly different greedy?
            # For now, just reset T.
            last_improve_time = time.time()
            
        r_idx = random.randint(0, n_rows - 1)
        # Bias towards earlier indices to allow more compression of the tail
        # j_idx=1 is very strong. j_idx=k is weaker.
        r = random.random()
        if r < 0.3:
            j_idx = 1
        elif r < 0.5:
            j_idx = 2
        elif r < 0.7:
            j_idx = random.randint(3, k)
        else:
            j_idx = random.randint(1, k)
            
        old_rows = copy.deepcopy(rows)
        old_used = used_diffs.copy()
        old_scope = get_scope(rows)
        
        apply_compress_move(r_idx, j_idx)
        
        new_scope = get_scope(rows)
        
        if new_scope < old_scope:
            best_rows = copy.deepcopy(rows)
            best_scope = new_scope
            last_improve_time = time.time()
        elif new_scope > old_scope:
            # Revert
            rows = old_rows
            used_diffs = old_used
        else:
            # Keep (explore)
            pass
            
        if T > 1.0:
            T *= cooling_rate
            
    return best_rows
