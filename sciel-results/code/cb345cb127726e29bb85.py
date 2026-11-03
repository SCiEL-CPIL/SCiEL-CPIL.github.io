import random
import time
import math

def _generate_initial_population(n_rows, k, pop_size, seed=42):
    """Generate initial population using randomized greedy construction."""
    rng = random.Random(seed)
    population = []
    for _ in range(pop_size):
        used = set()
        rows = []
        valid = True
        for _ in range(n_rows):
            row = [0]
            # Randomized greedy: try to pick small numbers, but with some randomness
            # to ensure diversity.
            # We pick the next number by trying candidates from 1 upwards,
            # but occasionally skipping to allow different structures.
            # For simplicity, let's just do a standard greedy with a random start offset or random choice among valid small candidates.
            x = 0
            # To ensure diversity, we can bias the choice.
            # Let's try candidates 1, 2, ... up to a limit, pick a random valid one from the first few valid ones.
            found = False
            # Limit search to reasonable scope for init
            max_val = 150 
            for candidate in range(1, max_val + 1):
                new_diffs = {candidate - a for a in row}
                if not (new_diffs & used):
                    # Accept with some probability to add diversity? 
                    # Or just take the first one (greedy) for most, random for some?
                    # Let's take the first valid one for 80% of rows, random valid in next 5 for 20%.
                    if rng.random() < 0.8:
                        row.append(candidate)
                        used |= new_diffs
                        found = True
                        break
                    else:
                        # Store as potential, but keep looking? 
                        # Simpler: just pick the first valid one always for init, diversity comes from mutation.
                        # Actually, let's just do pure greedy for init to ensure valid start.
                        row.append(candidate)
                        used |= new_diffs
                        found = True
                        break
            if not found:
                valid = False
                break
            rows.append(row)
        if valid:
            population.append(rows)
    return population

def _reconstruct_row(used_diffs, k, max_limit, rng):
    """
    Reconstruct a row of size k+1 starting at 0, increasing,
    such that its differences are disjoint from used_diffs.
    Minimizes the largest entry.
    Returns the row or None if not found within max_limit.
    """
    # Backtracking search
    # State: current row, last element
    # We want to minimize the last element.
    
    best_row = None
    best_max = max_limit + 1
    
    def backtrack(row, last_val):
        nonlocal best_row, best_max
        
        if len(row) == k + 1:
            if last_val < best_max:
                best_max = last_val
                best_row = row[:]
            return
        
        # Pruning: even if we add 1 for each remaining slot, 
        # is it possible to beat best_max?
        remaining = k + 1 - len(row)
        if last_val + remaining >= best_max:
            return
            
        # Try candidates for the next element
        # Start from last_val + 1
        # We can limit the upper bound of candidate to best_max - remaining
        # to ensure we can still complete the row below best_max
        
        upper_bound = best_max - remaining
        if upper_bound <= last_val:
            return
            
        # To speed up, we can iterate candidates in increasing order.
        # We stop if we find a valid completion, but we want the *minimal* max.
        # So we should find the smallest possible next element that leads to a valid solution.
        
        # Optimization: Check if there are enough available differences?
        # For now, simple backtracking.
        
        for candidate in range(last_val + 1, upper_bound + 1):
            # Check differences
            valid = True
            for a in row:
                d = candidate - a
                if d in used_diffs:
                    valid = False
                    break
            if not valid:
                continue
            
            # Also check internal uniqueness (implied by increasing and distinct diffs? 
            # No, a_j - a_i must be distinct. 
            # Since row is increasing, d > 0.
            # We need to ensure d is not already in the row's internal diffs.
            # We haven't stored internal diffs in backtrack state easily.
            # Let's compute internal diffs for the new row prefix.
            
            # Actually, since we build incrementally, we just need to ensure
            # (candidate - a) is not in used_diffs AND not equal to any 
            # (row[j] - row[i]) for i<j in the current row.
            
            # Let's precompute internal diffs of current row? 
            # Or just check against all pairs in row.
            # Row size is small (<=6), so O(k^2) is fine.
            
            for i in range(len(row)):
                d = candidate - row[i]
                if d in used_diffs:
                    valid = False
                    break
                # Check if d is already a difference within the current row
                # Since we are adding candidate as the last element,
                # the new differences are candidate - row[i].
                # We need to ensure these are distinct from each other?
                # Yes, candidate - row[i] for different i are distinct because row[i] are distinct.
                # We need to ensure they are distinct from existing differences in the row.
                # Existing differences are row[j] - row[i] for i<j in row.
                
                # Let's just collect all diffs of the new row and check uniqueness + disjointness.
                # This is O(k^2) per step. k=5, so 15 diffs. Very fast.
                
            if valid:
                # Re-check full validity of new row prefix + candidate
                new_row = row + [candidate]
                diffs = set()
                ok = True
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in used_diffs or d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                    if not ok:
                        break
                if ok:
                    backtrack(new_row, candidate)
                    # Since we want the minimal max, and we iterate candidate increasing,
                    # the first successful completion found at this depth will have the minimal 
                    # last element for this prefix? 
                    # Not necessarily, because a larger intermediate value might allow a smaller final value?
                    # No, the final value is the max. 
                    # If we find a valid completion with max M, we update best_max.
                    # We continue searching to see if we can find a smaller M.
                    # But since we iterate candidate increasing, and the final value is >= candidate,
                    # if we find a solution with final value F, any other solution from this branch 
                    # with a larger next candidate will have final value >= next_candidate > candidate.
                    # It *could* be smaller than F if F was very large? 
                    # No, F >= candidate. 
                    # If we pick a larger next candidate, the final value will be at least that larger candidate.
                    # So if we found a solution with max F, and we try a next candidate C > F, 
                    # the new max will be >= C > F. So it won't be better.
                    # Thus, we can prune candidates > best_max.
                    # And since we update best_max, the search space shrinks.
                    
                    # Important: We should return early if we found a solution 
                    # with max == candidate? 
                    # If candidate is the last element, then max is candidate.
                    # If len(new_row) == k+1, we updated best_max.
                    # If we are not at the last level, we continue.
                    
                    # To speed up, if we found a solution where the current candidate 
                    # is the final element (i.e., len(new_row)==k+1), 
                    # and candidate is the smallest possible next element, 
                    # is it the global minimum? 
                    # Not necessarily, because a different path earlier might lead to a smaller final element.
                    # But for this specific prefix, the first valid completion found 
                    # (with smallest next elements) yields the smallest final element 
                    # *for this prefix*. 
                    # We still need to explore other prefixes (other choices for earlier elements).
                    
                    pass # Continue backtracking to find better solutions

    backtrack([0], 0)
    
    if best_row:
        return best_row
    return None

def _optimize_row_in_context(rows, idx, rng):
    """
    Try to improve row `idx` in `rows` by reconstructing it
    to minimize its max value, given the other rows are fixed.
    Returns new rows if improved, else None.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    # Collect differences of all other rows
    used_diffs = set()
    for i in range(n_rows):
        if i == idx:
            continue
        row = rows[i]
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                used_diffs.add(row[l] - row[j])
                
    current_max = max(rows[idx])
    
    # Try to reconstruct with a limit slightly smaller than current_max
    # to ensure improvement. If we can't find one smaller, we might still find one equal?
    # We want to minimize. So we search for min possible.
    # We can set max_limit to current_max to see if we can do better.
    # Actually, _reconstruct_row finds the *minimum* possible max given the constraints.
    # So we just call it with a high limit (e.g., current_max) and see what it returns.
    # If it returns a row with max < current_max, we accept.
    
    new_row = _reconstruct_row(used_diffs, k, current_max, rng)
    
    if new_row and max(new_row) < current_max:
        new_rows = rows[:]
        new_rows[idx] = new_row
        return new_rows
    return None

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    start_time = time.time()
    time_limit = 110  # Keep 10s buffer
    
    # Parameters
    pop_size = 40
    generations = 200
    
    # Initialize
    population = _generate_initial_population(n_rows, k, pop_size, seed=42)
    
    if not population:
        # Fallback to simple greedy
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
        
    # Track best
    best_rows = min(population, key=lambda r: max(max(row) for row in r))
    best_score = max(max(row) for row in best_rows)
    
    rng = random.Random(42)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        # Elitism: keep the best 10%
        elite_size = max(1, pop_size // 10)
        sorted_pop = sorted(population, key=lambda r: max(max(row) for row in r))
        new_population.extend(sorted_pop[:elite_size])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            p1 = sorted_pop[rng.randrange(len(sorted_pop))]
            p2 = sorted_pop[rng.randrange(len(sorted_pop))]
            
            # Crossover: Randomly select rows from p1 or p2
            child = []
            for i in range(n_rows):
                if rng.random() < 0.5:
                    child.append(p1[i][:])
                else:
                    child.append(p2[i][:])
            
            # Mutation/Repair: Try to optimize each row
            # We pick a random subset of rows to optimize to save time
            # or optimize all rows? 
            # Optimizing all rows is O(n_rows * search_cost).
            # Search cost is backtracking. 
            # Let's try to optimize a random 2-3 rows per child.
            
            indices_to_optimize = rng.sample(range(n_rows), min(3, n_rows))
            for idx in indices_to_optimize:
                improved = _optimize_row_in_context(child, idx, rng)
                if improved:
                    child = improved
            
            # Validate child? 
            # The repair step ensures each optimized row is valid with respect to others *at the time of optimization*.
            # However, if we optimize row A, then row B, row B's optimization sees A's new state.
            # So the final state should be valid.
            # But we should double check validity to be safe.
            
            # Quick validity check
            valid = True
            seen = set()
            for row in child:
                if len(row) != k + 1 or row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    if i > 0 and row[i] <= row[i-1]:
                        valid = False
                        break
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in seen:
                            valid = False
                            break
                        seen.add(d)
                if not valid:
                    break
            
            if valid:
                new_population.append(child)
            else:
                # If invalid, try to fix by regenerating the whole child?
                # Or just discard.
                pass
        
        # Update population
        if len(new_population) < pop_size:
            # Fill with copies of best or random existing
            while len(new_population) < pop_size:
                new_population.append(population[rng.randrange(len(population))])
                
        population = new_population
        
        # Update best
        for ind in population:
            score = max(max(row) for row in ind)
            if score < best_score:
                best_score = score
                best_rows = ind[:]
                
    return best_rows
