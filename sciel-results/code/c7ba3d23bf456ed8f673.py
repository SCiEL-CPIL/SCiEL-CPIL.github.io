import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer for the evaluator
    
    # Number of restarts
    max_restarts = 200

    def get_flexibility(candidate_val, current_row, used_diffs, remaining_elements):
        """
        Estimate how many valid choices exist for the NEXT element after candidate_val.
        Higher is better (more flexible).
        """
        if remaining_elements == 0:
            return float('inf')
        
        count = 0
        last_val = candidate_val
        # Check a small window for the next element
        for y in range(last_val + 1, last_val + 40):
            new_diffs = {y - a for a in current_row + [candidate_val]}
            if not new_diffs & used_diffs:
                count += 1
                if count > 5:
                    break
        return count

    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        used_diffs = set()
        rows = []
        
        def build_row(row_idx):
            if len(rows) == n_rows:
                return True
            
            current_row = [0]
            
            def build_elements():
                if len(current_row) == k + 1:
                    rows.append(current_row[:])
                    return True
                
                last_val = current_row[-1]
                upper_bound = best_scope - 1 if best_scope < float('inf') else 200
                
                candidates = []
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_diffs:
                        continue
                    
                    rem = (k + 1) - (len(current_row) + 1)
                    flex = get_flexibility(x, current_row, used_diffs, rem)
                    candidates.append((flex, x, new_diffs))
                    
                    if len(candidates) > 100:
                        break
                
                if not candidates:
                    return False
                
                candidates.sort(key=lambda item: (-item[0], item[1]))
                
                top_n = min(5, len(candidates))
                top_candidates = candidates[:top_n]
                random.shuffle(top_candidates)
                
                for flex, x, new_diffs in top_candidates:
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if build_elements():
                        return True
                    
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
                    
                    if time.time() - start_time > time_limit:
                        return False
                
                for flex, x, new_diffs in candidates[top_n:]:
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if build_elements():
                        return True
                    
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
                    
                    if time.time() - start_time > time_limit:
                        return False
                        
                return False
            
            if not build_elements():
                return False
            
            return build_row(row_idx + 1)

        if build_row(0):
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break

    if best_result is None:
        # Fallback to deterministic greedy if random search failed (unlikely)
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result
