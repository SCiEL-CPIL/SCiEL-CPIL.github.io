import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (used >> (x - a)) & 1:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used |= (1 << (x - a))
                    row.append(x)
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)

    def solve_with_max(M, seed=None):
        """
        Returns a list of rows if a valid configuration with max entry <= M exists,
        otherwise returns None.
        """
        if seed is not None:
            random.seed(seed)
        
        used = 0
        rows = []
        
        def backtrack(row_idx, row):
            nonlocal used, rows
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                return True
            
            if len(row) == 0:
                rows.append([0])
                if backtrack(row_idx, [0]):
                    return True
                rows.pop()
            else:
                current_row = row
                last = current_row[-1]
                
                # Generate candidates
                candidates = list(range(last + 1, M + 1))
                
                # Pruning: check if we can fit remaining elements
                remaining_needed = (k + 1) - len(current_row) - 1
                if remaining_needed > 0:
                    # The smallest possible max value for the rest is x + remaining_needed
                    # So we only need x such that x + remaining_needed <= M
                    # i.e., x <= M - remaining_needed
                    max_x = M - remaining_needed
                    if max_x < last + 1:
                        return False
                    candidates = [x for x in candidates if x <= max_x]
                    
                # Shuffle candidates to add randomness
                random.shuffle(candidates)
                
                for x in candidates:
                    conflict = False
                    new_diffs = 0
                    for a in current_row:
                        d = x - a
                        if (used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if not conflict:
                        used |= new_diffs
                        current_row.append(x)
                        
                        if backtrack(row_idx, current_row):
                            return True
                        
                        current_row.pop()
                        used ^= new_diffs
                        
                return False
        
        rows = [[0]]
        if backtrack(0, [0]):
            return rows
        return None

    lower_bound = n_rows * (k * (k + 1) // 2)
    current_target = best_max - 1

    # Try to find smaller M
    # We will try each M with a few random restarts if it fails quickly?
    # Or just one attempt per M.
    
    while current_target >= lower_bound:
        if time.time() > deadline - 5.0:
            break
            
        sol = solve_with_max(current_target, seed=current_target)
        if sol:
            best_rows = sol
            best_max = current_target
            current_target -= 1
        else:
            # If it failed, it's likely infeasible.
            # But to be safe against false negatives from timeout, 
            # we can check if the time taken was close to the limit.
            # For now, we assume if it returns None, it's infeasible.
            break

    return best_rows
