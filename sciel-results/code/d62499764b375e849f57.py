<<<<<<< SEARCH
    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a boolean array for O(1) difference lookups.
        """
        rows = [r[:] for r in initial_rows]
        
        # Estimate max possible difference for array size
        # Conservative upper bound
        max_possible = max(max(r) for r in rows) + 100
        # Use a boolean array for used differences
        used_arr = bytearray(max_possible + 1)
        
        # Initialize used_arr
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        used_arr[d] = 1
        
        def compute_scope(rows):
            return max(max(r) for r in rows)
        
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Temperature schedule
        t_start = time.time()
        initial_temp = max(1.0, current_scope / 5.0)
        cooling_rate = 0.9995
        
        while time.time() - t_start < time_limit:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Bias toward the last element to encourage scope reduction
            if random.random() < 0.4:
                pos = k  # Last element
            else:
                pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            # Try to change this value
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 50
            
            if min_val > max_val:
                continue

            # For the last element, bias toward smaller values to reduce scope
            if pos == k and random.random() < 0.6:
                # Try values in [min_val, old_val] to shrink
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val)
                else:
                    continue
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue

            # Check if changing this value maintains validity
            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)

            # Temporarily remove old_diffs from used_arr
            for d in old_diffs:
                if d <= max_possible:
                    used_arr[d] = 0

            # Try adding new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d > max_possible:
                        valid = False
                        break
                    if used_arr[d] == 1:
                        valid = False
                        break
                    new_diffs.append(d)

            if not valid:
                # Restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                continue

            # Compute new scope
            new_row_last = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            other_max = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
            new_scope = max(other_max, new_row_last)

            # Accept or reject using smooth surrogate
            delta = current_scope - new_scope
            if delta > 0 or random.random() < math.exp(delta / max(initial_temp, 0.1)):
                # Accept
                rows[row_idx][pos] = new_val
                for d in new_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope
=======
    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a boolean array for O(1) difference lookups.
        Maintains per-row max for efficient scope computation.
        """
        rows = [r[:] for r in initial_rows]
        
        # Estimate max possible difference for array size
        max_possible = max(max(r) for r in rows) + 100
        # Use a boolean array for used differences
        used_arr = bytearray(max_possible + 1)
        
        # Initialize used_arr
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        used_arr[d] = 1
        
        # Maintain per-row max for efficient scope computation
        row_maxs = [max(r) for r in rows]
        
        def compute_scope():
            return max(row_maxs)
        
        current_scope = compute_scope()
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Temperature schedule
        t_start = time.time()
        initial_temp = max(1.0, current_scope / 5.0)
        cooling_rate = 0.9995
        
        while time.time() - t_start < time_limit:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Bias toward the last element to encourage scope reduction
            if random.random() < 0.4:
                pos = k  # Last element
            else:
                pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            # Try to change this value
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 50
            
            if min_val > max_val:
                continue

            # For the last element, bias toward smaller values to reduce scope
            if pos == k and random.random() < 0.6:
                # Try values in [min_val, old_val] to shrink
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val)
                else:
                    continue
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue

            # Check if changing this value maintains validity
            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)

            # Temporarily remove old_diffs from used_arr
            for d in old_diffs:
                if d <= max_possible:
                    used_arr[d] = 0

            # Try adding new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d > max_possible:
                        valid = False
                        break
                    if used_arr[d] == 1:
                        valid = False
                        break
                    new_diffs.append(d)

            if not valid:
                # Restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                continue

            # Compute new scope efficiently using row_maxs
            old_row_max = row_maxs[row_idx]
            if pos == k:
                new_row_max = new_val if new_val > old_row_max else old_row_max
                # Actually need to recompute since intermediate elements might be larger
                new_row_max = max(rows[row_idx][pos-1], new_val)
                # More precisely:
                new_row_max = max(rows[row_idx][:pos+1])
            else:
                new_row_max = old_row_max  # Changing an intermediate element doesn't change max unless it was the max
            
            # More careful: compute new row max
            if pos == k:
                new_row_max = max(rows[row_idx][:k]) if k > 0 else 0
                if new_val > new_row_max:
                    new_row_max = new_val
            else:
                new_row_max = old_row_max
            
            other_max = max(row_maxs[:row_idx] + row_maxs[row_idx+1:]) if n_rows > 1 else 0
            new_scope = max(other_max, new_row_max)

            # Accept or reject using smooth surrogate
            delta = current_scope - new_scope
            if delta > 0 or random.random() < math.exp(delta / max(initial_temp, 0.1)):
                # Accept
                rows[row_idx][pos] = new_val
                for d in new_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                
                # Update row_maxs
                if pos == k:
                    row_maxs[row_idx] = max(rows[row_idx][:k], new_val) if k > 0 else new_val
                else:
                    row_maxs[row_idx] = max(rows[row_idx])
                
                current_scope = compute_scope()

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope
>>>>>>> REPLACE

