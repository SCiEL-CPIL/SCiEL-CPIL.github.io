import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 110  # 10s margin from 120s limit
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return len(diffs) == len(set(diffs))
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def backtrack_row(used, max_val, row, depth):
        """Build one row via backtracking within max_val."""
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        # Try values in random order for diversity
        candidates = list(range(start, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            row.append(x)
            used_copy = used | new_diffs
            
            result = backtrack_row(used_copy, max_val, row, depth + 1)
            if result is not None:
                return result
            
            row.pop()
        
        return None
    
    def backtrack_construct(max_val, num_rows, timeout_check=None):
        """Try to construct all rows within max_val using backtracking."""
        used = set()
        rows = []
        
        for i in range(num_rows):
            if timeout_check and timeout_check():
                return None
            
            row = [0]
            result = backtrack_row(used, max_val, row, 1)
            if result is None:
                return None
            
            rows.append(result)
            for a in result:
                for b in result:
                    if a != b:
                        used.add(abs(a - b))
        
        return rows
    
    def local_search(rows, max_val, iterations=1000):
        """Try to improve rows by swapping elements."""
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        
        for _ in range(iterations):
            # Pick a random row and position
            ri = random.randint(0, len(best) - 1)
            pi = random.randint(1, k)  # positions 1..k (not position 0)
            
            old_val = best[ri][pi]
            
            # Try to replace with a different value
            new_val = random.randint(best[ri][pi - 1] + 1, min(max_val, best[ri][pi + 1] - 1)) if pi < k else random.randint(best[ri][pi - 1] + 1, max_val)
            
            if new_val == old_val:
                continue
            
            # Check if new value creates conflicts
            test_row = best[ri][:]
            test_row[pi] = new_val
            
            # Check within-row uniqueness
            if len(set(test_row)) != len(test_row):
                continue
            
            # Check all diffs
            all_diffs = []
            valid = True
            for r in best:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.append(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check new row's diffs don't conflict
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in all_diffs:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Accept if it improves or maintains scope
            new_scope = max(max(r) for r in best if r is not test_row)
            new_scope = max(new_scope, max(test_row))
            
            if new_scope <= best_scope:
                best[ri] = test_row
                best_scope = new_scope
        
        return best, best_scope
    
    # Start with greedy
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # Try to find solutions at progressively lower scopes
    # Theoretical lower bound from counting: need total_diffs distinct diffs in [1, M]
    # So M >= total_diffs, but geometric constraints make it larger
    lower_bound = max(total_diffs, 50)  # conservative lower bound
    
    # Binary search style: try scopes from high to low
    current_max = best_scope
    
    while time.time() < deadline:
        # Try to find a solution with scope < current_max
        target_scope = current_max - 1
        
        if target_scope < lower_bound:
            break
        
        # Try backtracking with this max_val
        timeout_check = lambda: time.time() > deadline - 5
        
        result = backtrack_construct(target_scope, n_rows, timeout_check)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            current_max = target_scope
        else:
            # If backtracking fails, try a few more times with randomization
            found = False
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                result = backtrack_construct(target_scope, n_rows, timeout_check)
                if result is not None:
                    best_rows = result
                    best_scope = target_scope
                    current_max = target_scope
                    found = True
                    break
            
            if not found:
                # Give up on this target, try one lower
                current_max = target_scope
    
    # Local search to polish
    if time.time() < deadline - 5:
        polished, polished_scope = local_search(best_rows, best_scope, iterations=2000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
