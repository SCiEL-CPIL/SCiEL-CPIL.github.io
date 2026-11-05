import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    def generate_rows(limit, count):
        """Generate 'count' valid rows with max entry <= limit."""
        rows = []
        
        def backtrack(row, used_diffs, current_max_limit):
            if time.time() - start_time > time_limit - 10:
                return
            if len(row) == k + 1:
                if max(row) <= current_max_limit:
                    rows.append(row[:])
                return
            
            last = row[-1]
            candidates = []
            for x in range(last + 1, current_max_limit + 1):
                new_diffs = [x - a for a in row]
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                if any(d in used_diffs for d in new_diffs):
                    continue
                candidates.append(x)
            
            if not candidates:
                return

            random.shuffle(candidates)
            if len(candidates) > 20:
                candidates = random.sample(candidates, 20)

            for x in candidates:
                row.append(x)
                new_used = used_diffs.union([x - a for a in row[:-1]])
                backtrack(row, new_used, current_max_limit)
                row.pop()
                
                if len(rows) >= count:
                    return

        current_limit = 105
        while len(rows) < count and current_limit < 150:
            backtrack([0], set(), current_limit)
            if len(rows) < count:
                current_limit += 5
        
        return rows

    def select_disjoint_rows(pool, n_rows):
        """Select n_rows disjoint rows from pool to minimize max scope."""
        best_selection = None
        best_max_scope = float('inf')
        
        pool.sort(key=lambda r: max(r))
        limited_pool = pool[:200]
        
        def backtrack(idx, used_diffs, current_rows, current_max):
            nonlocal best_selection, best_max_scope
            
            if time.time() - start_time > time_limit - 5:
                return
                
            if len(current_rows) == n_rows:
                if current_max < best_max_scope:
                    best_max_scope = current_max
                    best_selection = [list(r) for r in current_rows]
                return
            
            if current_max > best_max_scope:
                return

            for i in range(idx, len(limited_pool)):
                row = limited_pool[i]
                diffs = row_diffs(row)
                
                if diffs & used_diffs:
                    continue
                    
                new_max = max(current_max, max(row))
                if new_max > best_max_scope:
                    continue
                
                backtrack(i + 1, used_diffs.union(diffs), current_rows + [row], new_max)
                
                if time.time() - start_time > time_limit - 5:
                    return

        backtrack(0, set(), [], 0)
        return best_selection

    best_rows_found = None
    best_score = float('inf')
    
    pool = generate_rows(120, 500)
    
    if pool:
        selection = select_disjoint_rows(pool, n_rows)
        if selection:
            best_rows_found = selection
            best_score = max(max(r) for r in selection)
    
    if best_rows_found is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows_found = rows
        best_score = max(max(r) for r in best_rows_found)
        
    if time.time() - start_time < time_limit - 10:
        def local_search(rows):
            current_max = max(max(r) for r in rows)
            improved = True
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                for i in range(n_rows):
                    for pos in range(len(rows[i]) - 1, 0, -1):
                        row = rows[i]
                        current_val = row[pos]
                        min_val = row[pos-1] + 1
                        for v in range(min_val, current_val):
                            new_row = row[:]
                            new_row[pos] = v
                            new_diffs = row_diffs(new_row)
                            if len(new_diffs) != expected_diffs:
                                continue
                            
                            conflict = False
                            for j in range(n_rows):
                                if i == j:
                                    continue
                                other_diffs = row_diffs(rows[j])
                                if new_diffs & other_diffs:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                rows[i] = new_row
                                if max(max(r) for r in rows) < current_max:
                                    current_max = max(max(r) for r in rows)
                                    improved = True
                                break
            return rows

        best_rows_found = local_search(best_rows_found)
        best_score = max(max(r) for r in best_rows_found)

    return best_rows_found
