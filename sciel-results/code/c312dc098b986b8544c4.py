import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial parameters
    # The task is (7, 5).
    # But the function takes n_rows and k.
    # We should handle general n_rows, k but optimize for the expected case.
    
    # If k is large, SA might be slow.
    # For k=5, it's fine.
    
    # Helper to calculate conflicts and scope
    def get_metrics(rows):
        diff_counts = {}
        max_val = 0
        for row in rows:
            for x in row:
                if x > max_val:
                    max_val = x
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for c in diff_counts.values():
            if c > 1:
                conflicts += c - 1
        return conflicts, max_val

    # Initialize rows
    # Start with a greedy-like initialization to be close to a solution?
    # Or random?
    # Random in [0, M_max]
    
    def random_row(m_max):
        # Generate k+1 distinct integers in [0, m_max]
        # Including 0
        if k == 0:
            return [0]
        # Choose k distinct non-zero integers
        # To ensure they are distinct and in range
        # If m_max is small, this might fail
        if m_max < k:
            # Fallback
            return list(range(k+1))
        
        # Random sample
        # We need 0 and k other distinct numbers
        # Sample from 1..m_max
        try:
            others = random.sample(range(1, m_max + 1), k)
        except ValueError:
            # Not enough numbers
            others = list(range(1, k+1))
        
        row = [0] + sorted(others)
        return row

    # We will try to find a solution with M_max = 112 first
    # If n_rows, k are different, we need to adjust M_max.
    # Counting bound: n_rows * k(k+1)/2
    bound = n_rows * (k * (k + 1) // 2)
    
    # Start M_max at max(bound + 10, 112) if it's the (7,5) case, else bound + 20
    if n_rows == 7 and k == 5:
        initial_m_max = 112
    else:
        initial_m_max = bound + 20
        
    current_m_max = initial_m_max
    best_scope = float('inf')
    best_rows = None
    
    # Run SA
    # We will run multiple rounds, decreasing M_max if we find a solution.
    
    while time.time() < deadline:
        # Initialize state for this round
        rows = [random_row(current_m_max) for _ in range(n_rows)]
        # Ensure 0 is first
        for i in range(n_rows):
            rows[i] = [0] + sorted(rows[i][1:])
            # Check if distinct
            if len(set(rows[i])) < k + 1:
                 # Retry init for this row
                 rows[i] = random_row(current_m_max)
                 rows[i] = [0] + sorted(rows[i][1:])
        
        conflicts, scope = get_metrics(rows)
        
        # If we already have a valid solution with smaller scope, we might want to keep it
        if conflicts == 0 and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            # Try to shrink
            current_m_max = scope - 1
            if current_m_max < bound:
                break # Can't go lower than bound
        
        # SA Loop
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling = 0.999
        
        # Local state for SA
        cur_conflicts = conflicts
        cur_scope = scope
        cur_rows = [r[:] for r in rows]
        
        # Pre-calculate difference counts for fast update?
        # For simplicity, recalculate if k is small.
        # k=5 -> 15 diffs per row. 7 rows -> 105 diffs.
        # Recalculating is fast enough.
        
        while time.time() < deadline:
            # Move
            r_idx = random.randint(0, n_rows - 1)
            # Pick a mark to change
            # We can't change the 0 mark easily if we want to keep it 0?
            # The problem says "starting at 0". So a_0 is always 0.
            # So we only change a_1 ... a_k.
            if k > 0:
                j_idx = random.randint(1, k) # 1 to k inclusive
            else:
                j_idx = 0 # Only 0, can't change
            
            old_val = cur_rows[r_idx][j_idx]
            
            # Generate new value
            # Bias towards current value? Or uniform?
            # Uniform in [0, current_m_max]
            new_val = random.randint(0, current_m_max)
            
            # Update row
            cur_rows[r_idx][j_idx] = new_val
            # Sort
            cur_rows[r_idx].sort()
            
            # Check for duplicates
            if len(set(cur_rows[r_idx])) < k + 1:
                # Revert
                cur_rows[r_idx][j_idx] = old_val
                cur_rows[r_idx].sort()
                # Or just continue to next iteration
                continue
            
            # Calculate new metrics
            new_conflicts, new_scope = get_metrics(cur_rows)
            
            # Note: new_scope might exceed current_m_max?
            # No, we picked new_val in [0, current_m_max], and other values were already <= current_m_max.
            # So new_scope <= current_m_max.
            
            delta_e = (new_conflicts - cur_conflicts) + 0.1 * (new_scope - cur_scope)
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                cur_conflicts = new_conflicts
                cur_scope = new_scope
                # cur_rows is already updated
                
                if cur_conflicts == 0 and cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [r[:] for r in cur_rows]
                    # Update current_m_max for future restarts?
                    # We can update it immediately to allow shrinking
                    # But we are in the middle of SA.
                    # Let's just record it.
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
                # If stuck, restart SA with new random init?
                # Or break and do a new round with lower M_max if we found a solution.
                # Let's check time and break to do a new round.
                break

        # End of SA round
        # If we found a solution, we can try to shrink M_max
        if best_rows is not None:
            # We found a valid solution.
            # Let's set current_m_max to best_scope - 1 for the next round
            # to see if we can find a better one.
            if current_m_max > best_scope - 1:
                current_m_max = best_scope - 1
            if current_m_max < bound:
                break
        else:
            # No solution found yet.
            # Maybe increase M_max?
            # Or just restart with same M_max.
            pass

    if best_rows is None:
        # Fallback to greedy if SA failed completely
        # This should not happen for (7,5) with 110s
        used, rows = set(), []
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

    return best_rows
