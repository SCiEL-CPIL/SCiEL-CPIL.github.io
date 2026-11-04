"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    # Greedy baseline
    def greedy_solution(seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Backtracking solver for a given M
    def backtrack_solve(M, time_limit, random_order=False):
        start = time.time()
        used = 0
        rows = []
        
        def build_row(global_used):
            if time.time() - start > time_limit:
                return None
            
            row = [0]
            row_used = 0
            
            def rec(pos, last_val):
                if time.time() - start > time_limit:
                    return None
                
                if pos == row_len:
                    # Row complete, check global
                    if row_used & global_used:
                        return None
                    # Recurse to next row
                    if len(rows) == n_rows - 1:
                        rows.append(list(row))
                        return True
                    else:
                        res = build_row(global_used | row_used)
                        if res:
                            rows.append(list(row))
                            return True
                        else:
                            rows.pop() if rows else None
                            return None
                
                # Generate candidates
                candidates = list(range(last_val + 1, M + 1))
                if random_order:
                    random.shuffle(candidates)
                
                for v in candidates:
                    new_diffs = 0
                    conflict = False
                    for u in row:
                        d = v - u
                        if (row_used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    if new_diffs & row_used:
                        continue
                    
                    row.append(v)
                    res = rec(pos + 1, v)
                    if res:
                        return True
                    row.pop()
                
                return None
            
            return rec(1, 0)
        
        return build_row(0)
    
    # Local search using bitmask
    def local_search(rows, deadline_t, max_iter=200):
        rows = [list(r) for r in rows]
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used |= (1 << (row[j] - row[i]))
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(rows[row_idx])
            
            # Remove old row's diffs
            old_diffs = 0
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs |= (1 << (old_row[j] - old_row[i]))
            used_without = used & ~old_diffs
            
            # Try to rebuild this row with smaller max
            target_max = old_row[-1] - 1
            if target_max < k:
                target_max = k
            
            improved = False
            for attempt in range(20):
                new_row = [0]
                new_diffs = 0
                success = True
                for pos in range(1, row_len):
                    # Try candidates in random order, preferring smaller values
                    candidates = list(range(new_row[-1] + 1, target_max + 1))
                    random.shuffle(candidates)
                    # Sort by value but with some randomness
                    candidates.sort(key=lambda x: (x, random.random()))
                    
                    found = False
                    for v in candidates:
                        nd = 0
                        conflict = False
                        for u in new_row:
                            d = v - u
                            if (new_diffs >> d) & 1:
                                conflict = True
                                break
                            if (used_without >> d) & 1:
                                conflict = True
                                break
                            nd |= (1 << d)
                        
                        if not conflict:
                            new_row.append(v)
                            new_diffs |= nd
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if success and len(new_row) == row_len and new_row[-1] < old_row[-1]:
                    # Verify no conflicts
                    if not (new_diffs & used_without):
                        # Check internal uniqueness
                        internal_ok = True
                        all_d = []
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                all_d.append(new_row[j] - new_row[i])
                        if len(set(all_d)) != len(all_d):
                            internal_ok = False
                        
                        if internal_ok:
                            rows[row_idx] = new_row
                            used = used_without | new_diffs
                            new_max = get_max_entry(rows)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = [list(r) for r in rows]
                            improved = True
                            break
                else:
                    # Rollback if we modified rows
                    rows[row_idx] = old_row
                    used = used_without | old_diffs
            
            if not improved:
                # Try swapping elements between two rows
                if n_rows >= 2:
                    i1, i2 = random.sample(range(n_rows), 2)
                    r1 = list(rows[i1])
                    r2 = list(rows[i2])
                    
                    if len(r1) > 1 and len(r2) > 1:
                        p1 = random.randint(1, len(r1) - 1)
                        p2 = random.randint(1, len(r2) - 1)
                        
                        # Compute diffs
                        d1 = 0
                        for a in range(len(r1)):
                            for b in range(a + 1, len(r1)):
                                d1 |= (1 << (r1[b] - r1[a]))
                        d2 = 0
                        for a in range(len(r2)):
                            for b in range(a + 1, len(r2)):
                                d2 |= (1 << (r2[b] - r2[a]))
                        
                        other = used & ~d1 & ~d2
                        
                        # Swap
                        r1[p1], r2[p2] = r2[p2], r1[p1]
                        
                        # Check increasing
                        if all(r1[a] < r1[a+1] for a in range(len(r1)-1)) and \
                           all(r2[a] < r2[a+1] for a in range(len(r2)-1)):
                            nd1 = 0
                            valid1 = True
                            for a in range(len(r1)):
                                for b in range(a + 1, len(r1)):
                                    d = r1[b] - r1[a]
                                    if (nd1 >> d) & 1:
                                        valid1 = False
                                        break
                                    nd1 |= (1 << d)
                            if valid1:
                                nd2 = 0
                                valid2 = True
                                for a in range(len(r2)):
                                    for b in range(a + 1, len(r2)):
                                        d = r2[b] - r2[a]
                                        if (nd2 >> d) & 1:
                                            valid2 = False
                                            break
                                        nd2 |= (1 << d)
                                if valid2 and not (nd1 & nd2) and not (nd1 & other) and not (nd2 & other):
                                    rows[i1] = r1
                                    rows[i2] = r2
                                    used = other | nd1 | nd2
                                    new_max = get_max_entry(rows)
                                    if new_max < best_max:
                                        best_max = new_max
                                        best_rows = [list(r) for r in rows]
                                    improved = True
            
            if not improved:
                # Restore
                rows[row_idx] = old_row
                used = used_without | old_diffs
        
        return best_rows, best_max
    
    # Main search
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    search_deadline = deadline - 5.0
    
    # Try backtracking with decreasing M values
    m_values = list(range(total_diffs, best_max + 1))
    random.shuffle(m_values)
    m_values.sort()  # Try smallest M first
    
    found = False
    for M in m_values[:min(15, len(m_values))]:
        if time.time() > search_deadline:
            break
        
        remaining = search_deadline - time.time()
        time_slice = min(remaining / 10, 5.0)
        time_slice = max(time_slice, 1.0)
        
        try:
            rows = backtrack_solve(M, time_slice, random_order=True)
            if rows and len(rows) == n_rows:
                if validate(rows):
                    curr_max = get_max_entry(rows)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = rows
                        found = True
                        break
        except Exception:
            pass
    
    # Local search to improve
    if time.time() < search_deadline:
        try:
            improved_rows, improved_max = local_search(best_rows, search_deadline, max_iter=100)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # More greedy restarts with local search
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            seed = restart * 7919 + 13
            g_rows = greedy_solution(seed=seed)
            g_max = get_max_entry(g_rows)
            if g_max < best_max:
                best_max = g_max
                best_rows = g_rows
            if time.time() < deadline - 5:
                try:
                    ir, im = local_search(g_rows, deadline - 5, max_iter=30)
                    if im < best_max:
                        best_max = im
                        best_rows = ir
                except Exception:
                    pass
    
    # Final validation
    if not validate(best_rows):
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
