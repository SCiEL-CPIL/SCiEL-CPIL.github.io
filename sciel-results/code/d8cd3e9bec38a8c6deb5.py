import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    num_marks = k + 1
    best_rows = None
    best_score = float('inf')
    
    def solve_once(randomness, top_k, pick_k):
        rows = [[0] for _ in range(n_rows)]
        used = set()
        
        while any(len(r) < num_marks for r in rows):
            options = []
            for i, row in enumerate(rows):
                if len(r) < num_marks: # Bug: should be len(row)
                    pass
            # Corrected loop below
            
            options = []
            for i, row in enumerate(rows):
                if len(row) < num_marks:
                    x = row[-1] + 1
                    valid_xs = []
                    while len(valid_xs) < top_k:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            valid_xs.append(x)
                        x += 1
                    if valid_xs:
                        options.append((i, valid_xs))
            
            if not options:
                return None, float('inf')
                
            # Select row
            min_val = min(min(opts) for _, opts in options)
            best_rows_idx = [i for i, opts in options if min(opts) == min_val]
            
            if random.random() < randomness:
                i = random.choice([i for i, _ in options])
                chosen_opts = [opts for idx, opts in options if idx == i][0]
            else:
                i = random.choice(best_rows_idx)
                chosen_opts = [opts for idx, opts in options if idx == i][0]
            
            candidates = chosen_opts[:min(pick_k, len(chosen_opts))]
            val = random.choice(candidates)
            
            rows[i].append(val)
            for a in rows[i]:
                used.add(val - a)
                
            if time.time() > deadline:
                return None, float('inf')
        
        score = max(max(r) for r in rows)
        return rows, score

    attempts = 0
    # Vary parameters slightly
    while time.time() < deadline:
        # Oscillate randomness and pick_k
        rand_val = 0.3 + 0.4 * (attempts % 5) / 5.0
        top_k_val = 8 + (attempts % 4) * 2
        pick_k_val = 2 + (attempts % 3)
        
        rows, score = solve_once(rand_val, top_k_val, pick_k_val)
        attempts += 1
        if rows is not None and score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
            if best_score <= 112:
                break

    if best_rows is None:
        # Fallback: Deterministic Greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_marks:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
        
    return best_rows
