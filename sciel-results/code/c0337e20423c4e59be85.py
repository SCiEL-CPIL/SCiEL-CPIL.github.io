import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    # Initial greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    while time.time() < deadline:
        # Randomized Greedy
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find first valid x
                x_min = row[-1] + 1
                while True:
                    new_diffs = {x_min - a for a in row}
                    if not new_diffs & used:
                        break
                    x_min += 1
                
                # Collect candidates
                candidates = []
                x = x_min
                # Limit candidate search to keep it fast
                # 20 candidates is a good balance
                while len(candidates) < 20:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Weighted random choice
                # Weights: 1, 0.5, 0.33...
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                x = random.choices(candidates, weights=weights, k=1)[0]
                
                row.append(x)
                # Calculate diffs for the chosen x
                # Note: we need to recalculate or store them
                # Since we checked it in the loop, we can just recompute for update
                used.update({x - a for a in row[:-1]})
            
            if not success:
                break
            rows.append(row)
        
        if success:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 105:
                    break

    return best_rows
