import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def is_sidon(arr):
        """Check if arr (sorted, distinct) is a Sidon set using vectorized differences."""
        n = len(arr)
        if n < 2:
            return True
        a = np.array(arr)
        # Compute all pairwise differences (i < j)
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(a[j] - a[i])
        diffs = np.array(diffs)
        if len(diffs) != len(np.unique(diffs)):
            return False
        return True

    def has_all_diffs(arr):
        """Check if all d in 1..30 appear as a difference."""
        n = len(arr)
        a = np.array(arr)
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                d = a[j] - a[i]
                if 1 <= d <= 30:
                    diffs.add(d)
        return len(diffs) == 30

    def get_all_diffs(arr):
        """Get set of all positive differences."""
        n = len(arr)
        a = np.array(arr)
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(a[j] - a[i])
        return diffs

    def score(arr):
        """
        Score function: lower is better.
        We want to minimize max(arr), but also need to satisfy constraints.
        If not Sidon or missing diffs, add large penalties.
        """
        if not is_sidon(arr):
            # Penalty for collisions: count duplicate differences
            n = len(arr)
            a = np.array(arr)
            diffs = []
            for i in range(n):
                for j in range(i+1, n):
                    diffs.append(a[j] - a[i])
            diffs = np.array(diffs)
            n_collisions = len(diffs) - len(np.unique(diffs))
            return 10**9 + n_collisions * 1000 + max(arr)
        
        # Sidon is satisfied, now check coverage
        n = len(arr)
        a = np.array(arr)
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                d = a[j] - a[i]
                if 1 <= d <= 30:
                    diffs.add(d)
        missing = 30 - len(diffs)
        if missing > 0:
            return 10**6 + missing * 100 + max(arr)
        
        # All constraints satisfied
        return max(arr)

    def greedy_construct():
        """Fast greedy to get a valid starting point."""
        a = [0]
        def get_diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in get_diffs(sorted(a)):
                continue
            # Try to add a single element
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                # Add two elements far away
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    best_solution = greedy_construct()
    best_score = score(best_solution)
    
    # Simulated Annealing
    current = list(best_solution)
    current_score = best_score
    
    n = len(current)
    max_val = max(current)
    
    t = 10000.0
    t_min = 1.0
    cooling = 0.9995
    
    while time.time() < deadline and t > t_min:
        # Perturbation
        new_current = current[:]
        
        # Choose a random element to move
        idx = random.randint(0, n - 1)
        if idx == 0:
            # 0 must stay 0? The task says "including 0". 
            # If we move 0, we must ensure 0 is in the set.
            # Let's keep index 0 as 0 to simplify, or swap it.
            # Actually, "including 0" means 0 must be in the set.
            # It doesn't mean the smallest must be 0, but since all are >= 0, it is.
            # If we move 0, we might lose 0. 
            # Strategy: Keep 0 fixed. Perturb indices 1..n-1.
            idx = random.randint(1, n - 1)
        
        # Move element to a random nearby position
        # Or swap with another element?
        # Let's try moving one element to a random value in [0, max_val + 100]
        # Or better: local move.
        step_size = int(0.1 * max_val) + 1
        delta = random.randint(-step_size, step_size)
        new_val = new_current[idx] + delta
        
        if new_val < 0:
            new_val = 0
        if new_val > 10**6:
            new_val = 10**6
            
        # Ensure distinctness
        if new_val in new_current:
            # If collision, try a small adjustment or skip
            # For simplicity, if collision, revert this step
            # Or find next available
            found_spot = False
            for check in range(1, 10):
                alt = new_val + check
                if alt not in new_current and alt >= 0 and alt <= 10**6:
                    new_val = alt
                    found_spot = True
                    break
            if not found_spot:
                for check in range(1, 10):
                    alt = new_val - check
                    if alt not in new_current and alt >= 0 and alt <= 10**6:
                        new_val = alt
                        found_spot = True
                        break
            if not found_spot:
                continue # Skip this iteration
        
        new_current[idx] = new_val
        new_current.sort()
        
        # Ensure 0 is in the set
        if 0 not in new_current:
            # This shouldn't happen if we kept index 0 fixed, but just in case
            # If 0 was moved, we broke the rule.
            # Since we forced idx != 0, and 0 was at index 0, 0 is still there.
            pass

        new_score = score(new_current)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < np.exp(-delta_score / t):
            current = new_current
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_solution = current[:]
        
        t *= cooling
        
    # Final polish: try to reduce the largest element by removing it if possible
    # or by local search with tighter constraints
    if time.time() < deadline:
        # Try to remove the largest element if the set still satisfies constraints
        # This would reduce the diameter
        if len(best_solution) > 1:
            trial = best_solution[:-1]  # Remove largest
            if 0 in trial and is_sidon(trial) and has_all_diffs(trial):
                best_solution = trial
                best_score = max(best_solution)
        
        # Try to reduce the largest element by moving it closer
        if len(best_solution) > 1:
            current = best_solution[:]
            current_score = best_score
            n = len(current)
            
            # Focus on the largest element
            idx = n - 1
            max_val = current[idx]
            
            # Try to move it to a smaller value
            for delta in range(1, int(0.2 * max_val) + 1):
                new_val = max_val - delta
                if new_val <= current[idx - 1]:
                    break
                if new_val in current:
                    continue
                trial = current[:]
                trial[idx] = new_val
                trial.sort()
                if is_sidon(trial) and has_all_diffs(trial):
                    best_solution = trial
                    best_score = max(best_solution)
                    max_val = new_val
                    current = trial[:]
                    current_score = best_score
                    break
    
    return best_solution
# EVOLVE-BLOCK-END
