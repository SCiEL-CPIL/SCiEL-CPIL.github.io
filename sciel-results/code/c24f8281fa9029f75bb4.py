import time
import random
import math
import sys


def _row_diffs_set(row):
    """Return set of all positive differences within a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_set(row))
    return len(all_diffs) == len(set(all_diffs))


def _global_mask(rows):
    """Compute global bitmask of all differences."""
    m = 0
    for row in rows:
        m |= _row_mask(row)
    return m


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy construction: build rows one by one, picking the smallest valid value
    for each position with some randomization.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            if max_scope_hint:
                high = max_scope_hint
            else:
                high = 200
            
            # Try values from low to high with randomization
            start = row[-1] + 1
            # Shuffle a window of candidates to add randomness
            window_size = min(30, high - start + 1)
            if window_size > 1:
                candidates = list(range(start, min(start + window_size, high + 1)))
                rng.shuffle(candidates)
                # Add some higher candidates too
                for v in range(start + window_size, high + 1, 3):
                    candidates.append(v)
            else:
                candidates = [start] if start <= high else []
            
            for val in candidates:
                if val <= row[-1]:
                    continue
                new_diffs = {val - prev for prev in row}
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                
                # Check within-row consistency
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue
                
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            
            if not found:
                # Fallback: try all values up to a larger bound
                for val in range(start, high + 20):
                    new_diffs = {val - prev for prev in row}
                    new_mask = 0
                    for d in new_diffs:
                        new_mask |= (1 << d)
                    if new_mask & used_mask:
                        continue
                    existing_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            existing_diffs.add(row[j] - row[i])
                    if new_diffs & existing_diffs:
                        continue
                    row.append(val)
                    used_mask |= new_mask
                    found = True
                    break
                if not found:
                    # Last resort: just add a large value
                    row.append(row[-1] + 50)
                    new_diffs = {row[-1] - prev for prev in row[:-1]}
                    for d in new_diffs:
                        used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows


def _fitness(rows):
    """
    Fitness: number of duplicate differences (lower is better).
    Returns (num_conflicts, scope).
    """
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_set(row))
    num_conflicts = len(all_diffs) - len(set(all_diffs))
    return num_conflicts, _scope(rows)


def _local_search(rows, num_elements, rng, time_limit, max_iter=50000):
    """
    Local search: try to reduce conflicts and scope by moving elements.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_conflicts, current_scope = _fitness(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_conflicts = current_conflicts
    best_scope = current_scope
    
    global_mask, row_masks = _global_mask(current_rows), [_row_mask(r) for r in current_rows]
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    stagnation = 0
    max_stagnation = 2000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 5
        
        if low >= high:
            continue
        
        # Bias toward lower values
        if old_val > low + 2 and rng.random() < 0.8:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Tentatively apply
        row[pos] = new_val
        
        # Check within-row uniqueness
        diffs_in_row = set()
        for i in range(n_elem):
            ri = row[i]
            for j in range(i + 1, n_elem):
                d = row[j] - ri
                if d in diffs_in_row:
                    row[pos] = old_val
                    break
                diffs_in_row.add(d)
        else:
            new_row_mask = 0
            for d in diffs_in_row:
                new_row_mask |= (1 << d)
            
            other_mask = global_mask & ~row_masks[row_idx]
            if new_row_mask & other_mask:
                row[pos] = old_val
            else:
                # Accept
                row_masks[row_idx] = new_row_mask
                global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
                # Actually need to recompute: remove old, add new
                global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
                # Wait, row_masks[row_idx] was already updated above. Let me redo.
                # Actually the logic is: global_mask = (global_mask without old row mask) | new row mask
                # But we already updated row_masks[row_idx]. Let me just recompute global.
                # For simplicity, recompute global mask occasionally.
                
                new_conflicts, new_scope = _fitness(current_rows)
                if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_scope < best_scope):
                    best_conflicts = new_conflicts
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    stagnation = 0
                else:
                    stagnation += 1
                # Update global mask
                global_mask = _global_mask(current_rows)
                row_masks = [_row_mask(r) for r in current_rows]
        # End else
        
    return best_rows, best_conflicts, best_scope


def _crossover_swap(rows_a, rows_b, rng, num_elements):
    """
    Row-pair swap crossover: exchange the difference sets between two rows.
    This is the key structural novelty to test whether recombination helps.
    
    We create children by swapping the "tails" (all elements except 0) between
    two rows, then attempting to repair conflicts via local adjustment.
    """
    # Simple swap: exchange the entire row content (except 0)
    child_a = [0] + rows_b[1:]
    child_b = [0] + rows_a[1:]
    
    # Repair: ensure strictly increasing and reduce conflicts
    for child in [child_a, child_b]:
        for i in range(1, len(child)):
            if child[i] <= child[i-1]:
                child[i] = child[i-1] + 1
    
    return child_a, child_b


def _genetic_algorithm(rows, num_elements, rng, time_limit, pop_size=8, gen_limit=200):
    """
    Island-model genetic algorithm.
    """
    start = time.time()
    
    # Initialize population
    population = [r[:] for r in rows]
    # Add some random variations
    for i in range(1, pop_size):
        if time.time() > time_limit:
            break
        var_rows = [r[:] for r in rows]
        # Mutate a few elements
        for _ in range(3):
            row_idx = rng.randint(0, len(var_rows) - 1)
            pos = rng.randint(1, num_elements - 1)
            row = var_rows[row_idx]
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val + 20
            if low < high:
                row[pos] = rng.randint(low, high)
        population.append(var_rows)
    
    # Evaluate initial population
    pop_fitness = [_fitness(r) for r in population]
    best_idx = min(range(len(population)), key=lambda i: (pop_fitness[i][0], pop_fitness[i][1]))
    best_rows = [r[:] for r in population[best_idx]]
    best_conflicts, best_scope = pop_fitness[best_idx]
    
    gen = 0
    while gen < gen_limit and time.time() < time_limit:
        gen += 1
        
        # Evaluate all
        pop_fitness = [_fitness(r) for r in population]
        
        # Select parents (tournament)
        def tournament_select(pop, fit, k=3):
            indices = rng.sample(range(len(pop)), k)
            return min(indices, key=lambda i: (fit[i][0], fit[i][1]))
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best
        best_idx = min(range(len(population)), key=lambda i: (pop_fitness[i][0], pop_fitness[i][1]))
        new_population.append([r[:] for r in population[best_idx]])
        
        while len(new_population) < pop_size:
            if time.time() > time_limit:
                break
            
            # Pick two parents
            p1 = tournament_select(population, pop_fitness)
            p2 = tournament_select(population, pop_fitness)
            
            # Crossover with some probability
            if rng.random() < 0.5:
                # Swap crossover
                child1, child2 = _crossover_swap(population[p1], population[p2], rng, num_elements)
                # Repair conflicts via local adjustment
                for child in [child1, child2]:
                    # Ensure increasing
                    for i in range(1, len(child)):
                        if child[i] <= child[i-1]:
                            child[i] = child[i-1] + 1
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            else:
                # Just copy with mutation
                child = [r[:] for r in population[p1]]
                # Mutate
                for _ in range(rng.randint(1, 3)):
                    row_idx = rng.randint(0, len(child) - 1)
                    pos = rng.randint(1, num_elements - 1)
                    row = child[row_idx]
                    old_val = row[pos]
                    low = row[pos - 1] + 1
                    high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val + 10
                    if low < high:
                        row[pos] = rng.randint(low, high)
                new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Update best
        pop_fitness = [_fitness(r) for r in population]
        for i, (conf, scope) in enumerate(pop_fitness):
            if conf < best_conflicts or (conf == best_conflicts and scope < best_scope):
                best_conflicts = conf
                best_scope = scope
                best_rows = [r[:] for r in population[i]]
    
    return best_rows, best_conflicts, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Generate initial solutions
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple greedy constructions with different seeds
    for seed in range(15):
        if time.time() > DEADLINE - 60:
            break
        rng_local = random.Random(seed * 13 + 7)
        rows = _greedy_construct(n_rows, num_elements, rng_local)
        if _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    
    # If no valid solution found, use a simple sequential fallback
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search refinement
    if time.time() < DEADLINE - 40:
        improved_rows, improved_conf, improved_scope = _local_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 50
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 3: Genetic algorithm with swap crossover
    if time.time() < DEADLINE - 30:
        ga_rows, ga_conf, ga_scope = _genetic_algorithm(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20,
            pop_size=6,
            gen_limit=150
        )
        if ga_scope < best_scope:
            best_scope = ga_scope
            best_rows = ga_rows
    
    # Phase 4: Final local search polish
    if time.time() < DEADLINE - 10:
        improved_rows, improved_conf, improved_scope = _local_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 5
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
