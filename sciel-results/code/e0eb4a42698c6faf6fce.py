import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def beam_reconstruct_row(used, k, beam_width=15, max_candidates_per_beam=30, max_limit=None):
        beams = [([0], 0)]
        
        for step in range(1, k + 1):
            new_beams = []
            for row, _ in beams:
                last = row[-1]
                x = last
                found = 0
                while found < max_candidates_per_beam:
                    x += 1
                    if max_limit is not None and x > max_limit:
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        new_row = row + [x]
                        new_beams.append((new_row, x))
                        found += 1
            if not new_beams:
                return None
            new_beams.sort(key=lambda x: x[1])
            beams = new_beams[:beam_width]
            
        if not beams:
            return None
        return beams[0][0]

    # 1. Initial Greedy
    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local Search
    T = 50.0
    min_T = 0.1
    cooling = 0.99
    
    while time.time() - start_time < deadline:
        i = random.randint(0, n_rows - 1)
        old_row = best_rows[i]
        
        used_without_i = set()
        for j, r in enumerate(best_rows):
            if j != i:
                used_without_i |= get_diffs(r)
        
        new_row = beam_reconstruct_row(used_without_i, k, beam_width=15, max_candidates_per_beam=30, max_limit=best_scope)
        
        if new_row is None:
            # If no row found within limit, try without limit?
            # Or just skip.
            continue
            
        new_diffs = get_diffs(new_row)
        if new_diffs & used_without_i:
            continue
        
        best_rows[i] = new_row
        new_scope = max(max(r) for r in best_rows)
        
        if new_scope < best_scope:
            best_scope = new_scope
            T = 50.0
        elif new_scope == best_scope:
            pass
        else:
            delta = new_scope - best_scope
            prob = math.exp(-delta / T)
            if random.random() >= prob:
                best_rows[i] = old_row
                T *= cooling
                continue
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_rows
