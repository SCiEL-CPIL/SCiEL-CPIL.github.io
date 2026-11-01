import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140
    random.seed(789)
    
    n_elem = k + 1
    diffs_per_row = n_elem * (n_elem - 1) // 2
    
    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i+1, n):
                s.add(row[j] - ri)
        return s
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != n_elem:
                return False
            if row[0] != 0:
                return False
            for i in range(1, n_elem):
                if row[i] <= row[i-1]:
                    return False
            rd = row_diffs_set(row)
            if rd & used:
                return False
            used |= rd
        return True
    
    def find_best_row(forbidden, n_elem, time_limit=2.0):
        """
        Find a row of n_elem increasing integers starting at 0
        whose differences don't intersect forbidden, minimizing max entry.
        """
        start_time = time.time()
        lower = n_elem - 1
        upper = 300
        
        for max_val in range(lower, upper + 1):
            if time.time() - start_time > time_limit:
                break
            row = [0]
            local_used = set()
            
            def dfs():
                if time.time() - start_time > time_limit:
                    return False
                if len(row) == n_elem:
                    return True
                last = row[-1]
                for x in range(last + 1, max_val + 1):
                    new_diffs = {x - a for a in row}
                    if new_diffs & local_used:
                        continue
                    if new_diffs & forbidden:
                        continue
                    row.append(x)
                    local_used.update(new_diffs)
                    if dfs():
                        return True
                    row.pop()
                    local_used.difference_update(new_diffs)
                return False
            
            if dfs():
                return row[:]
        
        return None
    
    def build_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = find_best_row(used, n_elem, time_limit=3.0)
            if row is None:
                # Fallback to greedy
                row = [0]
                while len(row) < n_elem:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                        x += 1
            else:
                used |= row_diffs_set(row)
            rows.append(row)
        return rows
    
    # Phase 1: Initial construction
    rows = build_initial()
    best = [r[:] for r in rows]
    best_val = max_entry(best)
    
    # Phase 2: Row-wise optimization
    for iteration in range(50):
        if time.time() > deadline - 30:
            break
        improved = False
        for i in range(n_rows):
            if time.time() > deadline - 30:
                break
            forbidden = set()
            for j in range(n_rows):
                if j != i:
                    forbidden |= row_diffs_set(rows[j])
            
            new_row = find_best_row(forbidden, n_elem, time_limit=1.0)
            if new_row is not None:
                temp_rows = [r[:] for r in rows]
                temp_rows[i] = new_row
                temp_max = max_entry(temp_rows)
                if temp_max < best_val:
                    rows = temp_rows
                    best = [r[:] for r in rows]
                    best_val = temp_max
                    improved = True
        
        if not improved:
            break
    
    # Phase 3: Local search
    cur = [r[:] for r in best]
    cur_val = best_val
    row_diffs_list = [row_diffs_set(r) for r in cur]
    cur_used = set()
    for rd in row_diffs_list:
        cur_used |= rd
    
    T = 10.0
    T_min = 0.05
    cool = 0.99999
    
    no_improve = 0
    max_no_improve = 200000
    
    while time.time() < deadline - 5:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, n_elem)
        old_val = cur[ri][pi]
        
        candidates = []
        for d in range(-6, 7):
            if d == 0:
                continue
            nv = old_val + d
            if 0 < nv < 300:
                candidates.append(nv)
        if old_val == cur_val:
            for d in range(1, min(old_val, 12)):
                candidates.append(old_val - d)
        for _ in range(2):
            candidates.append(random.randint(1, max(old_val + 10, 30)))
        
        random.shuffle(candidates)
        moved = False
        
        for new_val in candidates:
            if new_val < 1:
                continue
            if pi > 0 and new_val <= cur[ri][pi - 1]:
                continue
            if pi < n_elem - 1 and new_val >= cur[ri][pi + 1]:
                continue
            
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            if any(cand_row[j] <= cand_row[j-1] for j in range(1, n_elem)):
                continue
            
            new_diffs = row_diffs_set(cand_row)
            old_diffs = row_diffs_list[ri]
            
            other_used = cur_used - old_diffs
            if new_diffs & other_used:
                continue
            
            other_max = 0
            for r in range(n_rows):
                if r != ri:
                    m = max(cur[r])
                    if m > other_max:
                        other_max = m
            cand_max = max(other_max, max(cand_row))
            
            if cand_max < cur_val:
                accept = True
            elif cand_max == cur_val:
                accept = random.random() < 0.15
            else:
                delta_e = cand_max - cur_val
                accept = random.random() < (1.0 / (1.0 + T * 0.02)) * 0.005
            
            if accept:
                cur_used.difference_update(old_diffs)
                cur_used.update(new_diffs)
                row_diffs_list[ri] = new_diffs
                cur[ri] = cand_row
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        T = max(T_min, T * cool)
        
        if no_improve > max_no_improve:
            random.seed(random.randint(0, 1000000))
            cur = build_initial()
            cur_val = max_entry(cur)
            row_diffs_list = [row_diffs_set(r) for r in cur]
            cur_used = set()
            for rd in row_diffs_list:
                cur_used |= rd
            no_improve = 0
            T = 10.0
    
    if not is_valid(best):
        best = build_initial()
    
    return best
