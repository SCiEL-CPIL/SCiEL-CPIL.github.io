import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def randomized_construct(seed):
        """
        Attempt to construct a valid set using randomized backtracking/greedy.
        Returns (rows, max_entry) or (None, inf) if failed.
        """
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            if time.time() > deadline:
                return None, float('inf')
            
            row = [0]
            success = True
            
            for _ in range(k):
                if time.time() > deadline:
                    return None, float('inf')
                
                candidates = []
                x = row[-1] + 1
                # Limit the search for the next number to avoid infinite loops
                limit = row[-1] + 100 
                
                while x < limit:
                    # Calculate new differences
                    new_diffs = set()
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        candidates.append((x, new_diffs))
                        # Stop collecting after a few candidates to add randomness
                        if len(candidates) >= 3:
                            break
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick a candidate
                # Bias towards smaller x to keep scope low, but allow randomness
                candidates.sort(key=lambda t: t[0])
                
                # With probability 0.7, pick the smallest. With 0.3, pick a random one from the first 3.
                if rng.random() < 0.7:
                    chosen_idx = 0
                else:
                    chosen_idx = rng.randint(0, len(candidates) - 1)
                
                x, diffs = candidates[chosen_idx]
                row.append(x)
                used_diffs.update(diffs)
            
            if not success:
                return None, float('inf')
            
            rows.append(row)
        
        max_entry = max(max(r) for r in rows)
        return rows, max_entry

    # Phase 1: Find a good initial solution via multiple randomized constructions
    seeds = list(range(200))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 15.0: # Leave 15s for SA
            break
            
        rows, scope = randomized_construct(seed)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            
    if best_rows is None:
        # Fallback to deterministic greedy if randomized failed
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - y for y in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = max(max(r) for r in rows)

    # Phase 2: Simulated Annealing to improve the scope
    if time.time() < deadline - 5.0:
        current_solution = [r[:] for r in best_rows]
        current_scope = best_scope
        
        def compute_cost(rows):
            scope = max(max(r) for r in rows)
            all_diffs = set()
            penalty = 0
            for row in rows:
                row_diffs = get_row_diffs(row)
                overlap = all_diffs & row_diffs
                penalty += len(overlap) * 1000
                all_diffs |= row_diffs
            return scope + penalty

        current_cost = compute_cost(current_solution)
        temperature = 100.0
        cooling_rate = 0.995
        min_temperature = 0.01
        
        while time.time() < deadline - 2.0 and temperature > min_temperature:
            # Generate a move
            new_solution = [r[:] for r in current_solution]
            
            # Pick a random row and position
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = new_solution[row_idx][pos]
            
            # Strategy: Try to reduce the value.
            min_val = new_solution[row_idx][pos - 1] + 1
            max_val = old_val
            
            if min_val >= max_val:
                temperature *= cooling_rate
                continue
            
            # Try a random value in the range, biased towards smaller
            # Use exponential distribution for delta?
            # Or just uniform.
            new_val = random.randint(min_val, max_val)
            
            # With some probability, try the absolute minimum valid value
            if random.random() < 0.2:
                new_val = min_val
            
            new_solution[row_idx][pos] = new_val
            
            # Validate the new solution
            row = new_solution[row_idx]
            if row[pos] <= row[pos - 1] or (pos < k and row[pos] >= row[pos + 1]):
                temperature *= cooling_rate
                continue
            
            all_diffs = set()
            valid = True
            for i, r in enumerate(new_solution):
                r_diffs = get_row_diffs(r)
                if all_diffs & r_diffs:
                    valid = False
                    break
                all_diffs |= r_diffs
            
            if not valid:
                temperature *= cooling_rate
                continue
            
            new_cost = compute_cost(new_solution)
            
            # Metropolis criterion
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / temperature):
                current_solution = new_solution
                current_cost = new_cost
                
                new_scope = max(max(r) for r in current_solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_solution]
            
            temperature *= cooling_rate
    
    return best_rows
