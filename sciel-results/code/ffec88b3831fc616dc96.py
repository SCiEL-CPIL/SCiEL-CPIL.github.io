"""Improved difference-triangle set constructor with randomized local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # --- Helper functions ---
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            d = get_diffs(row)
            if d & seen:
                return False
            seen |= d
        return True

    def max_entry(rows):
        """Get the maximum entry across all rows."""
        return max(max(row) for row in rows)

    # --- Greedy construction (baseline) ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Randomized greedy construction ---
    def randomized_greedy(rng, bias=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    # With some probability, skip this candidate to explore
                    if bias > 0 and rng.random() < bias:
                        continue
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Local search: swap two entries in a row ---
    def local_search(rows, rng, max_iters=1000):
        """Try to reduce the max entry by swapping entries."""
        best = rows[:]
        best_scope = max_entry(best)
        current_scope = best_scope
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and try to reduce its max
            r_idx = rng.randint(0, n_rows - 1)
            row = best[r_idx][:]
            
            # Try removing the largest element and finding a smaller replacement
            if len(row) < 2:
                continue
            
            max_val = row[-1]
            if max_val <= 0:
                continue
            
            # Try to replace max_val with a smaller value
            found = False
            for new_val in range(1, max_val):
                # Check if new_val can replace max_val
                candidate_row = row[:-1] + [new_val]
                candidate_row.sort()
                
                # Check strictly increasing
                if any(candidate_row[i] >= candidate_row[i+1] for i in range(len(candidate_row)-1)):
                    continue
                
                # Check that 0 is still first
                if candidate_row[0] != 0:
                    continue
                
                # Check differences don't conflict
                new_diffs = get_diffs(candidate_row)
                # Remove old diffs
                old_diffs = get_diffs(row)
                
                # Check against all other rows
                conflict = False
                for i, other_row in enumerate(best):
                    if i == r_idx:
                        continue
                    other_diffs = get_diffs(other_row)
                    if new_diffs & other_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    best[r_idx] = candidate_row
                    new_scope = max_entry(best)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                    found = True
                    break
            
            if not found:
                # Try a different approach: shift an intermediate element
                if len(row) > 2:
                    pos = rng.randint(1, len(row) - 2)
                    old_val = row[pos]
                    for new_val in range(row[pos-1] + 1, row[pos+1] if pos+1 < len(row) else old_val):
                        if new_val == old_val:
                            continue
                        candidate_row = row[:]
                        candidate_row[pos] = new_val
                        candidate_row.sort()
                        
                        if candidate_row[0] != 0:
                            continue
                        if any(candidate_row[i] >= candidate_row[i+1] for i in range(len(candidate_row)-1)):
                            continue
                        
                        new_diffs = get_diffs(candidate_row)
                        conflict = False
                        for i, other_row in enumerate(best):
                            if i == r_idx:
                                continue
                            other_diffs = get_diffs(other_row)
                            if new_diffs & other_diffs:
                                conflict = True
                                break
                        
                        if not conflict:
                            best[r_idx] = candidate_row
                            new_scope = max_entry(best)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                if current_scope < best_scope:
                                    best_scope = current_scope
                            break
        
        return best

    # --- Simulated annealing-like local search ---
    def anneal(rows, rng, max_iters=5000):
        """Annealing: accept worse moves with probability to escape local minima."""
        import math
        
        current = rows[:]
        current_scope = max_entry(current)
        best = current[:]
        best_scope = current_scope
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random row
            r_idx = rng.randint(0, n_rows - 1)
            row = current[r_idx][:]
            
            if len(row) < 2:
                continue
            
            # Generate a perturbation
            pos = rng.randint(1, len(row) - 1)
            delta = rng.randint(-5, 5)
            new_val = row[pos] + delta
            
            if new_val <= row[pos-1] or (pos + 1 < len(row) and new_val >= row[pos+1]):
                continue
            
            candidate_row = row[:]
            candidate_row[pos] = new_val
            candidate_row.sort()
            
            if candidate_row[0] != 0:
                continue
            
            # Check validity
            new_diffs = get_diffs(candidate_row)
            valid = True
            for i, other_row in enumerate(current):
                if i == r_idx:
                    continue
                other_diffs = get_diffs(other_row)
                if new_diffs & other_diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_scope = max_entry([current[i] if i != r_idx else candidate_row for i in range(n_rows)])
            
            if new_scope <= current_scope:
                current[r_idx] = candidate_row
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best = [r[:] for r in current]
            else:
                # Accept with probability exp(-delta/T)
                delta = new_scope - current_scope
                if delta < T:
                    prob = math.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        current[r_idx] = candidate_row
                        current_scope = new_scope
            
            T = max(T * cooling, T_min)
        
        return best

    # --- Main search loop ---
    rng = random.Random(42)
    
    # Start with greedy
    best_rows = greedy_construct()
    best_scope = max_entry(best_rows)
    
    # Try to improve with local search and annealing
    while time.time() < deadline:
        # Try local search
        candidate = local_search(best_rows, rng, max_iters=200)
        cand_scope = max_entry(candidate)
        if cand_scope < best_scope:
            best_rows = candidate
            best_scope = cand_scope
        
        # Try annealing
        candidate = anneal(best_rows, rng, max_iters=2000)
        cand_scope = max_entry(candidate)
        if cand_scope < best_scope:
            best_rows = candidate
            best_scope = cand_scope
        
        # Try randomized greedy restarts
        if time.time() < deadline - 30:
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                cand = randomized_greedy(rng, bias=0.1)
                cand_scope = max_entry(cand)
                if cand_scope < best_scope:
                    best_rows = cand
                    best_scope = cand_scope
                    # Improve with local search
                    improved = local_search(best_rows, rng, max_iters=500)
                    imp_scope = max_entry(improved)
                    if imp_scope < best_scope:
                        best_rows = improved
                        best_scope = imp_scope
    
    return best_rows
# EVOLVE-BLOCK-END
