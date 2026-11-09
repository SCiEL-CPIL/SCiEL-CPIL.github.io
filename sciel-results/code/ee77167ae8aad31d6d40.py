import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110

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

    # Get initial solution
    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # Simulated annealing
    rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Precompute used differences
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
    
    # Precompute max of other rows for each row
    # This is O(n^2 k^2) which is fine for small n, k
    # But we can update it dynamically
    
    temperature = 10.0
    cooling_rate = 0.99
    
    while time.time() < deadline and temperature > 0.1:
        # Pick a random row and a random index (1 to k)
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)
        old_val = rows[r_idx][i_idx]
        
        # Remove differences from this row
        row_diffs = set()
        for i in range(len(rows[r_idx])):
            for j in range(i+1, len(rows[r_idx])):
                row_diffs.add(rows[r_idx][j] - rows[r_idx][i])
        used -= row_diffs
        
        # Generate a random new value
        lower_bound = rows[r_idx][i_idx-1] + 1
        upper_bound = rows[r_idx][i_idx+1] - 1 if i_idx < k else old_val + 10
        # To allow increasing moves, we can go up to old_val + 10
        # But we want to minimize scope, so usually we try smaller values.
        # Let's try a range around old_val
        # To be more effective, try values that might reduce the scope
        # If i_idx == k, reducing new_val reduces the scope of this row
        # If i_idx < k, reducing new_val might not reduce the scope if the last mark is larger
        # So we should focus on the last mark (i_idx = k)
        # Let's bias the choice of i_idx towards k
        if random.random() < 0.5:
            i_idx = k
        
        new_val = random.randint(lower_bound, min(upper_bound, old_val + 5))
        
        # Check validity
        new_row = rows[r_idx][:]
        new_row[i_idx] = new_val
        new_row_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                new_row_diffs.add(new_row[j] - new_row[i])
        
        if not (new_row_diffs & used):
            # Valid move
            # Compute new scope
            new_row_scope = new_row[-1]
            # The overall scope is max(new_row_scope, max of other rows)
            # We can precompute the max of other rows
            other_max = max(max(r) for r in rows if r != r_idx)
            new_scope = max(new_row_scope, other_max)
            
            delta = new_scope - current_scope
            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                # Accept
                rows[r_idx] = new_row
                used |= new_row_diffs
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject
                used |= row_diffs
        else:
            # Invalid move, restore
            used |= row_diffs
        
        temperature *= cooling_rate
    
    return best_rows
