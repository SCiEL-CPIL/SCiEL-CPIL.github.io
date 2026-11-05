import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences between entries in the same row must be distinct over all rows.
    Minimize the largest entry.
    
    Uses a genetic algorithm approach:
    1. Generate initial population via greedy construction
    2. Evolve via crossover (swap entire rows) and mutation (shift-based repair)
    3. Select best individuals based on scope
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Precompute all possible pairs (i, j) with i < j for k+1 elements
    pairs = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    num_pairs = len(pairs)
    
    def get_row_diffs(row):
        """Return a frozenset of differences for a row."""
        return frozenset(row[j] - row[i] for i, j in pairs)
    
    def get_scope(rows):
        return max(max(r) for r in rows) if rows else 0
    
    def is_valid_rows(rows):
        """Check if all rows are valid and differences are distinct across rows."""
        used = set()
        for row in rows:
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            if row[0] != 0:
                return False
            diffs = get_row_diffs(row)
            if diffs & used:
                return False
            used |= diffs
        return True
    
    def greedy_construct(seed, max_scope_hint=None):
        """
        Greedy construction with a specific seed.
        """
        random.seed(seed)
        rows = []
        used = set()
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_scope_hint is not None and x > max_scope_hint:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used
    
    def normalize_row(row):
        """Shift row so it starts at 0."""
        if not row:
            return row
        offset = row[0]
        return [x - offset for x in row]
    
    def shift_row(row, shift):
        """Shift all elements in row by shift amount."""
        return [x + shift for x in row]
    
    def repair_row_conflicts(row, used_diffs, max_scope):
        """
        Try to find a valid position (shift) for row given used differences.
        Returns the best valid row or None.
        """
        base_diffs = get_row_diffs(row)
        
        # Try shifts from 0 upward
        for shift in range(0, max_scope + 1):
            candidate = shift_row(row, shift)
            if candidate[0] != 0:
                candidate = normalize_row(candidate)
            
            # Check if strictly increasing
            valid = True
            for i in range(1, len(candidate)):
                if candidate[i] <= candidate[i-1]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if max is within bounds
            if max(candidate) > max_scope:
                continue
            
            # Check difference conflicts
            cand_diffs = get_row_diffs(candidate)
            if cand_diffs & used_diffs:
                continue
            
            return candidate
        
        return None
    
    def repair_solution(rows, max_scope):
        """
        Repair a solution by greedily resolving conflicts.
        Returns repaired rows or None if failed.
        """
        repaired = []
        used = set()
        
        # Sort rows by their maximum value (try to place smaller rows first)
        row_indices = sorted(range(n_rows), key=lambda i: max(rows[i]))
        
        for idx in row_indices:
            row = rows[idx]
            repaired_row = repair_row_conflicts(row, used, max_scope)
            
            if repaired_row is None:
                # Try with a larger scope
                for extra in range(1, 20):
                    repaired_row = repair_row_conflicts(row, used, max_scope + extra)
                    if repaired_row is not None:
                        break
            
            if repaired_row is None:
                return None
            
            repaired.append(repaired_row)
            used |= get_row_diffs(repaired_row)
        
        return repaired
    
    def crossover(parent1, parent2):
        """
        Crossover: swap entire rows between parents.
        Returns two children.
        """
        child1_rows = []
        child2_rows = []
        
        for i in range(n_rows):
            if random.random() < 0.5:
                child1_rows.append(parent1[i].copy())
                child2_rows.append(parent2[i].copy())
            else:
                child1_rows.append(parent2[i].copy())
                child2_rows.append(parent1[i].copy())
        
        return child1_rows, child2_rows
    
    def mutate(rows, mutation_rate=0.3):
        """
        Mutation: apply small shifts to random rows and repair.
        """
        mutated = [r.copy() for r in rows]
        used = set()
        
        for r in mutated:
            used |= get_row_diffs(r)
        
        for i in range(n_rows):
            if random.random() < mutation_rate:
                # Try a small shift
                shift = random.choice([-2, -1, 1, 2])
                candidate = shift_row(mutated[i], shift)
                candidate = normalize_row(candidate)
                
                # Check if valid
                valid = True
                for j in range(1, len(candidate)):
                    if candidate[j] <= candidate[j-1]:
                        valid = False
                        break
                
                if valid:
                    old_diffs = get_row_diffs(mutated[i])
                    new_diffs = get_row_diffs(candidate)
                    other_diffs = used - old_diffs
                    
                    if not (new_diffs & other_diffs):
                        # Apply mutation
                        used = (used - old_diffs) | new_diffs
                        mutated[i] = candidate
        
        return mutated
    
    def genetic_algorithm(initial_population, time_budget):
        """
        Genetic algorithm to evolve the best solution.
        """
        # Initial population: list of (scope, rows)
        population = []
        for rows in initial_population:
            if is_valid_rows(rows):
                scope = get_scope(rows)
                population.append((scope, rows))
        
        if not population:
            return None, float('inf')
        
        # Sort by scope
        population.sort(key=lambda x: x[0])
        
        best_scope, best_rows = population[0]
        pop_size = min(20, len(population))
        current_pop = population[:pop_size]
        
        generation = 0
        no_improve = 0
        max_no_improve = 50
        
        while time.time() - start_time < time_budget:
            generation += 1
            new_pop = []
            
            # Elitism: keep top 2
            for i in range(min(2, len(current_pop))):
                new_pop.append(current_pop[i])
            
            # Generate offspring
            while len(new_pop) < pop_size:
                # Select parents via tournament
                def tournament_select():
                    candidates = random.sample(current_pop, min(3, len(current_pop)))
                    return min(candidates, key=lambda x: x[0])[1]
                
                p1 = tournament_select()
                p2 = tournament_select()
                
                # Crossover
                child1, child2 = crossover(p1, p2)
                
                # Mutate
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # Repair if needed
                if not is_valid_rows(child1):
                    repaired = repair_solution(child1, best_scope)
                    if repaired is not None:
                        child1 = repaired
                    else:
                        child1 = p1.copy()  # Keep parent if repair fails
                
                if not is_valid_rows(child2):
                    repaired = repair_solution(child2, best_scope)
                    if repaired is not None:
                        child2 = repaired
                    else:
                        child2 = p2.copy()
                
                # Add to new population
                if is_valid_rows(child1):
                    new_pop.append((get_scope(child1), child1))
                if is_valid_rows(child2):
                    new_pop.append((get_scope(child2), child2))
            
            # Sort new population
            new_pop.sort(key=lambda x: x[0])
            current_pop = new_pop[:pop_size]
            
            # Update best
            if current_pop[0][0] < best_scope:
                best_scope = current_pop[0][0]
                best_rows = current_pop[0][1]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
            
            # Occasionally add fresh random individuals
            if generation % 10 == 0:
                for _ in range(3):
                    seed = random.randint(0, 1000000)
                    rows, _ = greedy_construct(seed, best_scope + 20)
                    if rows is not None and is_valid_rows(rows):
                        scope = get_scope(rows)
                        if scope < best_scope + 10:
                            current_pop.append((scope, rows))
                            current_pop.sort(key=lambda x: x[0])
                            current_pop = current_pop[:pop_size]
        
        return best_rows, best_scope
    
    # Phase 1: Generate diverse initial population
    initial_population = []
    seen_diffs = set()
    
    for seed in range(100):
        rows, used = greedy_construct(seed)
        if rows is not None and is_valid_rows(rows):
            # Check diversity: ensure not too similar to existing
            row_signature = tuple(tuple(r) for r in rows)
            if row_signature not in seen_diffs:
                seen_diffs.add(row_signature)
                initial_population.append(rows)
    
    if not initial_population:
        # Fallback
        rows = [[i * 1000 for i in range(k + 1)] for _ in range(n_rows)]
        return rows
    
    # Sort by scope
    initial_population.sort(key=lambda r: get_scope(r))
    
    best_rows = initial_population[0]
    best_scope = get_scope(best_rows)
    
    # Phase 2: Genetic Algorithm
    time_spent = time.time() - start_time
    remaining = time_limit - time_spent
    
    if remaining > 10:
        ga_rows, ga_scope = genetic_algorithm(initial_population[:20], start_time + remaining)
        if ga_rows is not None and ga_scope < best_scope:
            best_scope = ga_scope
            best_rows = ga_rows
    
    # Final validation
    if not is_valid_rows(best_rows):
        for rows in initial_population:
            if is_valid_rows(rows):
                return rows
        return [[i * 1000 for i in range(k + 1)] for _ in range(n_rows)]
    
    return best_rows

# EVOLVE-BLOCK-START
# (No additional code needed here as construct is defined above)
# EVOLVE-BLOCK-END
