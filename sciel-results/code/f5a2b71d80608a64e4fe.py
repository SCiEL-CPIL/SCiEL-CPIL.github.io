import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 135  # safety margin: 150s limit, 15s buffer
    
    def time_left():
        return deadline - time.time()
    
    def greedy_construct(n_rows, k, randomize=False):
        """Greedy construction with optional randomization for diversity."""
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                if randomize:
                    # Try a random window of candidates
                    window = random.randint(1, 30)
                    candidates = list(range(x + 1, x + window + 1))
                    random.shuffle(candidates)
                    found = False
                    for cand in candidates:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            x = cand
                            used |= new
                            found = True
                            break
                    if not found:
                        # Fallback to sequential
                        while True:
                            x += 1
                            new = {x - a for a in row}
                            if not new & used:
                                row.append(x)
                                used |= new
                                break
                else:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
            rows.append(row)
        return rows
    
    def get_all_diffs(rows):
        """Get set of all differences across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_solution(rows):
        """Check if solution is valid (all diffs distinct)."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def score_solution(rows):
        """Return the max entry (lower is better)."""
        return max(r[-1] for r in rows)
    
    def simulated_annealing(rows, n_rows, k, deadline, max_iterations=5000):
        """Simulated annealing to minimize max entry."""
        current = [list(r) for r in rows]
        current_score = score_solution(current)
        best = [list(r) for r in current]
        best_score = current_score
        
        # Temperature schedule
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Choose a random row to modify
            idx = random.randint(0, n_rows - 1)
            
            # Compute used diffs from other rows
            used_others = set()
            for i, row in enumerate(current):
                if i == idx:
                    continue
                for j in range(len(row)):
                    for l in range(j + 1, len(row)):
                        used_others.add(row[l] - row[j])
            
            # Try to find a new row for position idx
            old_row = current[idx]
            old_row_diffs = row_diffs(old_row)
            
            # Strategy: try to build a new row that avoids used_others
            # and has a smaller max if possible
            new_row = None
            
            # Method 1: Backtrack with small max limit
            max_limit = old_row[-1] - random.randint(1, 10)
            if max_limit > k:
                new_row = _backtrack_row_limited(used_others, k, max_limit, deadline)
            
            # Method 2: If method 1 fails, try greedy with randomization
            if new_row is None:
                # Try several random greedy attempts
                for _ in range(5):
                    attempt = _greedy_row_random(used_others, k, old_row[-1], deadline)
                    if attempt is not None:
                        new_row = attempt
                        break
            
            # Method 3: If still nothing, try small perturbation
            if new_row is None:
                # Perturb one element of the row
                if len(old_row) > 1:
                    pos = random.randint(1, len(old_row) - 1)
                    delta = random.choice([-3, -2, -1, 1, 2, 3])
                    candidate = old_row[pos] + delta
                    if candidate > old_row[pos - 1] and (pos == len(old_row) - 1 or candidate < old_row[pos + 1]):
                        test_row = list(old_row)
                        test_row[pos] = candidate
                        test_diffs = row_diffs(test_row)
                        if not (test_diffs & used_others):
                            new_row = test_row
            
            if new_row is None:
                continue
            
            new_row_diffs = row_diffs(new_row)
            
            # Check validity
            if new_row_diffs & used_others:
                continue
            
            # Compute new score
            current[idx] = new_row
            new_score = score_solution(current)
            
            # Acceptance criterion
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = [list(r) for r in current]
            else:
                # Simulated annealing acceptance
                delta = new_score - current_score
                if delta < 0 or random.random() < (1.0 / (1.0 + delta / T)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(r) for r in current]
                else:
                    # Reject: revert
                    current[idx] = old_row
            
            # Cool down
            T = max(T_min, T * cooling)
        
        return best, best_score
    
    def _backtrack_row_limited(used_others, k, max_val, deadline):
        """Find a row with max <= max_val using backtracking."""
        if time.time() > deadline:
            return None
        
        found = []
        count = 0
        max_search = 50000
        
        def bt(pos, current_row, current_diffs):
            nonlocal count
            if time.time() > deadline or count > max_search:
                return False
            
            if pos == k + 1:
                found.append(list(current_row))
                return True
            
            if pos == 0:
                current_row.append(0)
                if bt(pos + 1, current_row, current_diffs):
                    current_row.pop()
                    return True
                current_row.pop()
                return False
            
            start = current_row[-1] + 1
            end = min(max_val, start + 50)  # Limit search window
            
            for candidate in range(start, end + 1):
                count += 1
                if time.time() > deadline or count > max_search:
                    return False
                
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = candidate - a
                    if d in used_others or d in current_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    current_row.append(candidate)
                    current_diffs.update(new_diffs)
                    if bt(pos + 1, current_row, current_diffs):
                        current_diffs -= new_diffs
                        current_row.pop()
                        return True
                    current_diffs -= new_diffs
                    current_row.pop()
            
            return False
        
        bt(0, [], set())
        return found[0] if found else None
    
    def _greedy_row_random(used_others, k, max_val, deadline):
        """Try to build a row greedily with randomization, max <= max_val."""
        if time.time() > deadline:
            return None
        
        for attempt in range(3):
            row, x = [0], 0
            row_diffs_set = set()
            success = True
            
            while len(row) < k + 1:
                if time.time() > deadline:
                    return None
                
                # Try candidates in random order within a window
                window = random.randint(5, 20)
                candidates = list(range(x + 1, min(x + window + 1, max_val + 1)))
                random.shuffle(candidates)
                
                found = False
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used_others) and not (new_diffs & row_diffs_set):
                        row.append(cand)
                        row_diffs_set |= new_diffs
                        x = cand
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success and row[-1] <= max_val:
                return row
        
        return None
    
    # Phase 1: Generate diverse initial solutions
    initial_solutions = []
    
    # Deterministic greedy
    initial_solutions.append(greedy_construct(n_rows, k, randomize=False))
    
    # Randomized greedy variants
    for _ in range(4):
        initial_solutions.append(greedy_construct(n_rows, k, randomize=True))
    
    # Select best initial solution
    best_initial = min(initial_solutions, key=score_solution)
    best_rows = [list(r) for r in best_initial]
    best_score = score_solution(best_rows)
    
    # Phase 2: Run simulated annealing from multiple starts
    sa_budget = max(1, int(time_left() * 0.8))
    n_sa_runs = min(5, max(1, sa_budget // 20))
    
    for run in range(n_sa_runs):
        if time_left() < 15:
            break
        
        # Pick a starting solution
        if run == 0:
            start_rows = best_rows
        else:
            start_rows = initial_solutions[run % len(initial_solutions)]
        
        sa_deadline = deadline - 10
        improved_rows, improved_score = simulated_annealing(
            start_rows, n_rows, k, sa_deadline, max_iterations=3000
        )
        
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
    
    # Phase 3: Final local search polish
    if time_left() > 20:
        # Try to improve each row individually
        for _ in range(3):  # Multiple passes
            if time_left() < 15:
                break
            improved = False
            order = list(range(n_rows))
            random.shuffle(order)
            
            for idx in order:
                if time_left() < 15:
                    break
                
                used_others = set()
                for i, row in enumerate(best_rows):
                    if i == idx:
                        continue
                    for j in range(len(row)):
                        for l in range(j + 1, len(row)):
                            used_others.add(row[l] - row[j])
                
                old_row = best_rows[idx]
                old_max = old_row[-1]
                
                # Try to find a better row
                new_row = _backtrack_row_limited(used_others, k, old_max - 1, deadline)
                if new_row is None:
                    new_row = _greedy_row_random(used_others, k, old_max - 1, deadline)
                
                if new_row is not None and new_row[-1] < old_max:
                    best_rows[idx] = new_row
                    new_score = score_solution(best_rows)
                    if new_score < best_score:
                        best_score = new_score
                    improved = True
            
            if not improved:
                break
    
    # Final validity check
    if not is_valid_solution(best_rows):
        # Fallback to deterministic greedy
        best_rows = greedy_construct(n_rows, k, randomize=False)
    
    return best_rows
# EVOLVE-BLOCK-END
