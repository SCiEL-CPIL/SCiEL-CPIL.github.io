import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_solution = None
    best_max = float('inf')
    
    def is_valid_solution(rows):
        # This is for verification, but we maintain validity during construction
        # Just checking distinct differences
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def local_improve(rows):
        """
        Try to reduce the maximum value by local moves.
        Returns the improved rows.
        """
        improved = True
        while improved:
            improved = False
            # Find the row with the largest last element
            max_val = max(row[-1] for row in rows)
            max_row_idx = next(i for i, row in enumerate(rows) if row[-1] == max_val)
            row = rows[max_row_idx]
            
            # Try to reduce the largest element
            # We can try to reduce row[-1] by 1, 2, ...
            # Or reduce any element that is not 0
            # Let's try to reduce the largest element first
            last_idx = len(row) - 1
            current_val = row[last_idx]
            
            # Try decreasing current_val
            for new_val in range(current_val - 1, row[last_idx-1], -1):
                # Check if new_val is valid
                # Differences with other elements in the row must be distinct and not used elsewhere
                # We need a global set of used diffs from other rows
                other_diffs = set()
                for i, r in enumerate(rows):
                    if i == max_row_idx:
                        continue
                    for a in range(len(r)):
                        for b in range(a+1, len(r)):
                            other_diffs.add(r[b] - r[a])
                
                # Check differences within the modified row
                valid = True
                new_row = row.copy()
                new_row[last_idx] = new_val
                # Check order
                if new_val <= new_row[last_idx-1]:
                    continue
                
                row_diffs = set()
                for a in range(len(new_row)):
                    for b in range(a+1, len(new_row)):
                        d = new_row[b] - new_row[a]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check against other rows
                if row_diffs & other_diffs:
                    continue
                
                # Valid move!
                rows[max_row_idx] = new_row
                improved = True
                break # Restart the while loop as the max might have changed or another row is now max
            
            if improved:
                continue
                
            # If we couldn't reduce the max element, try reducing other elements in that row?
            # Or try other rows?
            # For simplicity, let's just try to reduce the global max.
            # If we can't reduce the max, we might be stuck.
            # We can try to reduce any element that allows a smaller max?
            # Actually, reducing a non-max element doesn't reduce the max immediately.
            # So we focus on the max element.
            
            # If we can't reduce the max element, break?
            # Maybe we can swap? No, swapping doesn't change values.
            # Maybe we can reduce a smaller element to allow the max to be reduced?
            # That's more complex. Let's stick to reducing the max element.
            if not improved:
                # Try to reduce any element in any row to see if it enables a future reduction?
                # This is too complex. Let's just break.
                pass
                
        return rows

    def greedy_construct(randomize=True):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                # Search for next x
                # Heuristic: look ahead a bit?
                # Just try x = last + 1, last + 2, ...
                # Limit search range?
                # The max value shouldn't be too huge.
                # Let's search up to last + 50 initially, then expand?
                # Or just iterate.
                
                # To allow randomization, we collect the first few valid candidates
                valid_candidates = []
                x = last + 1
                # We don't know the upper bound, so we iterate.
                # But to keep it fast, we might cap the search if we are in a restart.
                # However, if we cap too low, we might fail to construct.
                # Let's iterate without a hard cap, but break if we find enough candidates for randomization.
                
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(x)
                        if not randomize or len(valid_candidates) >= 5:
                            break
                    x += 1
                    # Safety break if x gets too large?
                    if x > 1000:
                        break
                
                if not valid_candidates:
                    return None # Failed to construct
                
                if randomize:
                    # Pick randomly from the candidates
                    x = random.choice(valid_candidates)
                else:
                    x = valid_candidates[0]
                
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Run multiple iterations
    num_iterations = 1000
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize
        random.seed(i) # Or just use random
        rows = greedy_construct(randomize=True)
        if rows is None:
            continue
        
        # Local Improvement
        rows = local_improve(rows)
        
        current_max = max(row[-1] for row in rows)
        if current_max < best_max:
            best_max = current_max
            best_solution = [r.copy() for r in rows]
            
            # If we reach the theoretical lower bound, we can stop?
            # Lower bound is roughly (n_rows * k * (k+1) / 2)
            # But it's not a tight bound.
            # If best_max is small, we can stop early?
            # Let's just run until time limit.

    if best_solution is None:
        # Fallback to deterministic greedy
        return greedy_construct(randomize=False)
        
    return best_solution
