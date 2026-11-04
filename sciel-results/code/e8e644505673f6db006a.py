"""Difference triangle set (7,5): minimize the largest entry.
Multi-objective evolutionary approach with difference-swap mutations."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.0))
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _crossover(parent1, parent2, k):
    """Merge partial row constructions from two parents.
    For each row position, take the prefix from one parent and complete
    with elements from the other, resolving conflicts."""
    n_rows = len(parent1)
    child = []
    used = set()
    
    # Shuffle row order to break symmetry
    row_order = list(range(n_rows))
    random.shuffle(row_order)
    
    for r_idx in row_order:
        # Choose which parent to base this row on
        base_parent = parent1[r_idx] if random.random() < 0.5 else parent2[r_idx]
        
        # Start with the base row's prefix
        row = [0]
        # Try to extend using elements from both parents
        candidates = []
        
        # Collect candidate values from both parents at this row position
        for p in [parent1[r_idx], parent2[r_idx]]:
            for v in p[1:]:
                if v not in candidates:
                    candidates.append(v)
        
        random.shuffle(candidates)
        
        # Greedily build the row using candidates, then fill gaps
        for v in candidates:
            if v <= row[-1]:
                continue
            new_diffs = {v - a for a in row}
            if new_diffs & used:
                continue
            row.append(v)
            used |= new_diffs
            if len(row) == k + 1:
                break
        
        # If we didn't fill the row, complete greedily
        if len(row) < k + 1:
            x = row[-1]
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
        
        child.append(row)
    
    # Restore original row order
    child_ordered = [None] * n_rows
    for new_idx, old_idx in enumerate(row_order):
        child_ordered[old_idx] = child[new_idx]
    
    return child_ordered


def _difference_swap_mutation(rows, k, used, num_swaps=None):
    """Perform difference-swap moves: replace an element with another value
    that resolves conflicts while trying to reduce the max entry."""
    if num_swaps is None:
        num_swaps = random.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = random.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        # Determine valid range for new value
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 15
        
        if low >= high:
            continue
        
        # Try to find a new value that reduces the max or is at least valid
        best_new_val = None
        best_score = float('inf')
        
        # Sample candidate values
        candidates = set()
        for _ in range(20):
            candidates.add(random.randint(low, high))
        # Also try values near the current max to potentially reduce it
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 6):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            if new_val < low or new_val > high:
                continue
            
            # Calculate old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            # Calculate new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check conflict
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            # Score: prefer reducing the max entry
            new_row_max = max(row[:pos] + [new_val] + row[pos+1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max  # Negative is better
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            # Apply the swap
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used


def _evolutionary_search(n_rows, k, deadline, pop_size=12):
    """Multi-objective evolutionary algorithm for difference triangle sets."""
    rng = random.Random(42)
    
    # Initialize population with diverse greedy solutions
    population = []
    for i in range(pop_size):
        rows = _greedy_construct(n_rows, k, randomize=(i > 0), seed_offset=i * 17)
        population.append(rows)
    
    best_rows = min(population, key=_scope)
    best_scope = _scope(best_rows)
    
    generation = 0
    stagnation = 0
    
    while time.time() < deadline - 5.0:
        generation += 1
        current_best = _scope(best_rows)
        
        # Create next generation
        next_gen = [list(best_rows)]  # Elitism
        
        # Crossover + Mutation
        while len(next_gen) < pop_size and time.time() < deadline - 5.0:
            # Select parents (tournament selection)
            p1_idx = rng.randint(0, len(population) - 1)
            p2_idx = rng.randint(0, len(population) - 1)
            
            # Crossover
            child = _crossover(population[p1_idx], population[p2_idx], k)
            
            # Compute used set
            used = _all_diffs(child)
            
            # Mutation (difference-swap)
            if time.time() < deadline - 5.0:
                child, used = _difference_swap_mutation(child, k, used)
            
            next_gen.append(child)
        
        # Evaluate and select
        scored = []
        for rows in next_gen:
            scope = _scope(rows)
            scored.append((scope, rows))
        
        scored.sort(key=lambda x: x[0])
        
        # Keep the best
        if scored[0][0] < best_scope:
            best_rows = scored[0][1]
            best_scope = scored[0][0]
            stagnation = 0
        else:
            stagnation += 1
        
        # Update population: keep top half, add some random restarts if stagnating
        population = [s[1] for s in scored[:pop_size // 2]]
        
        if stagnation > 3:
            # Add random restarts to escape local optima
            for _ in range(pop_size // 2):
                if time.time() < deadline - 5.0:
                    rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=rng.randint(0, 10000))
                    population.append(rows)
            stagnation = 0
        
        # Fill remaining slots with mutations of best
        while len(population) < pop_size and time.time() < deadline - 5.0:
            used = _all_diffs(best_rows)
            mutated, _ = _difference_swap_mutation(best_rows, k, used, num_swaps=random.randint(1, 4))
            population.append(mutated)
    
    return best_rows, best_scope


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Evolutionary search
    if time.time() < deadline - 10.0:
        evo_rows, evo_scope = _evolutionary_search(n_rows, k, deadline - 10.0, pop_size=10)
        if evo_scope < best_scope:
            best_rows = evo_rows
            best_scope = evo_scope
    
    # Phase 3: Final refinement with difference-swap on best
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        # Multiple rounds of difference-swap mutations
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=random.randint(2, 5))
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
