import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin for 90s limit

# Pre-compute indices for A+A and A*A (i <= j)
INDICES = np.triu_indices(MAX_N)

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers.
    A is a list or array of 20 ints.
    """
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[INDICES[0]] + A_arr[INDICES[1]]
    products = A_arr[INDICES[0]] * A_arr[INDICES[1]]
    
    # Count unique values
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_gp(start, ratio_num, ratio_den, n):
    """Generate n distinct integers approximating a geometric progression."""
    A = []
    val = float(start)
    ratio = float(ratio_num) / float(ratio_den)
    
    for _ in range(n):
        val_int = int(round(val))
        if val_int < 1:
            val_int = 1
        if val_int > MAX_VAL:
            break
        if val_int not in A:
            A.append(val_int)
        val *= ratio
        
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    
    return A[:n]

def _crossover(A1, A2, rng):
    """
    Structure-aware crossover: blend geometric structures.
    Sort both, then for each position pick from parent with probability,
    but bias toward preserving relative ordering.
    """
    A1s = sorted(A1)
    A2s = sorted(A2)
    
    child = []
    used = set()
    
    for i in range(MAX_N):
        if rng.random() < 0.5:
            val = A1s[i]
        else:
            val = A2s[i]
        
        # Ensure distinctness
        attempts = 0
        while val in used and attempts < 50:
            # Perturb
            delta = rng.randint(-max(1, val // 5), max(1, val // 5))
            val = max(1, min(MAX_VAL, val + delta))
            attempts += 1
        
        if val in used:
            # Force a new value
            val = rng.randint(1, MAX_VAL)
            while val in used:
                val = rng.randint(1, MAX_VAL)
        
        used.add(val)
        child.append(val)
    
    return child

def _mutate(A, rng, strength=1.0):
    """
    Mutate a candidate: replace 1-3 elements with new values.
    Mix of local perturbations and global jumps.
    """
    A = list(A)
    num_mutations = rng.randint(1, 3)
    A_set = set(A)
    
    for _ in range(num_mutations):
        idx = rng.randint(0, MAX_N - 1)
        old_val = A[idx]
        
        if rng.random() < 0.4:
            # Local perturbation
            scale = max(1, int(old_val * strength * 0.3))
            new_val = old_val + rng.randint(-scale, scale)
        elif rng.random() < 0.7:
            # Medium jump
            new_val = rng.randint(max(1, old_val // 3), min(MAX_VAL, old_val * 3))
        else:
            # Global jump
            new_val = rng.randint(1, MAX_VAL)
        
        new_val = max(1, min(MAX_VAL, new_val))
        
        if new_val in A_set:
            # Try to find a nearby unused value
            for offset in range(1, 20):
                for delta in [offset, -offset]:
                    candidate = new_val + delta
                    if 1 <= candidate <= MAX_VAL and candidate not in A_set:
                        new_val = candidate
                        break
                else:
                    continue
                break
        
        if new_val in A_set:
            continue
            
        A_set.remove(old_val)
        A_set.add(new_val)
        A[idx] = new_val
    
    return A

def _genetic_algorithm(initial_pop, time_budget, rng):
    """
    Genetic algorithm with elitism and tournament selection.
    """
    pop_size = len(initial_pop)
    population = [list(ind) for ind in initial_pop]
    objectives = [_objective(ind) for ind in population]
    
    best_idx = int(np.argmin(objectives))
    best_A = list(population[best_idx])
    best_obj = objectives[best_idx]
    
    start_time = time.time()
    gen = 0
    stagnation_count = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Check if we have time for at least a few more evaluations
        if elapsed > time_budget - 2.0:
            break
        
        gen += 1
        stagnation_count += 1
        
        # Sort population by objective
        sorted_indices = np.argsort(objectives)
        population = [population[i] for i in sorted_indices]
        objectives = [objectives[i] for i in sorted_indices]
        
        # Elitism: keep top 20%
        elite_size = max(1, pop_size // 5)
        new_population = [list(population[i]) for i in range(elite_size)]
        new_objectives = [objectives[i] for i in range(elite_size)]
        
        # Generate offspring to fill population
        while len(new_population) < pop_size:
            # Tournament selection
            t_size = min(3, pop_size)
            tournament_indices = rng.sample(range(pop_size), t_size)
            winner = min(tournament_indices, key=lambda i: objectives[i])
            parent1 = population[winner]
            
            # Second parent: sometimes random, sometimes from top half
            if rng.random() < 0.3:
                parent2 = rng.choice(population)
            else:
                top_half = max(1, pop_size // 2)
                parent2 = population[rng.randint(0, top_half - 1)]
            
            # Crossover
            if rng.random() < 0.7:
                child = _crossover(parent1, parent2, rng)
            else:
                child = list(parent1)
            
            # Mutation
            child = _mutate(child, rng, strength=1.0 / (1 + gen * 0.01))
            
            if not _is_valid(child):
                # Fix validity
                child_set = set(child)
                if len(child_set) != MAX_N:
                    # Remove duplicates
                    child = list(dict.fromkeys(child))
                    while len(child) < MAX_N:
                        val = rng.randint(1, MAX_VAL)
                        if val not in child_set:
                            child.append(val)
                            child_set.add(val)
                    child = child[:MAX_N]
                child = sorted(child)
            
            new_population.append(child)
            new_objectives.append(_objective(child))
        
        population = new_population
        objectives = new_objectives
        
        # Update best
        current_best_idx = int(np.argmin(objectives))
        if objectives[current_best_idx] < best_obj:
            best_obj = objectives[current_best_idx]
            best_A = list(population[current_best_idx])
            stagnation_count = 0
        
        # If stagnating for too long, inject diversity
        if stagnation_count > 30:
            # Inject a few random individuals
            for _ in range(3):
                if rng.random() < 0.5:
                    # Random GP
                    start = rng.randint(1, 50)
                    ratio = rng.uniform(1.1, 2.5)
                    gp = []
                    val = float(start)
                    for _ in range(MAX_N):
                        v = int(round(val))
                        if 1 <= v <= MAX_VAL and v not in gp:
                            gp.append(v)
                        val *= ratio
                    if len(gp) >= MAX_N:
                        population[rng.randint(0, pop_size-1)] = sorted(gp[:MAX_N])
                        objectives[rng.randint(0, pop_size-1)] = _objective(population[rng.randint(0, pop_size-1)])
                else:
                    # Random
                    pop_idx = rng.randint(0, pop_size-1)
                    population[pop_idx] = sorted(rng.sample(range(1, MAX_VAL+1), MAX_N))
                    objectives[pop_idx] = _objective(population[pop_idx])
            stagnation_count = 0
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    
    # Generate diverse initial population from parameterized families
    initial_pop = []
    
    # Family 1: Various geometric progressions
    for ratio_num, ratio_den in [(3,2), (5,4), (4,3), (7,5), (5,3), (6,5), (8,7), (9,8), (5,2), (3,1)]:
        for start in [1, 2, 3, 5]:
            gp = _generate_gp(start, ratio_num, ratio_den, MAX_N)
            if _is_valid(gp):
                initial_pop.append(sorted(gp))
    
    # Family 2: Arithmetic progressions
    for a, d in [(1,1), (1,2), (2,3), (5,7), (10,13), (1,3), (2,5)]:
        ap = [a + i*d for i in range(MAX_N)]
        if max(ap) <= MAX_VAL and _is_valid(ap):
            initial_pop.append(sorted(ap))
    
    # Family 3: Mixed structures
    initial_pop.append(sorted(list(range(1, 11)) + [100*(i+1) for i in range(10)]))
    initial_pop.append(sorted(list(range(1, 6)) + [10**i for i in range(1, 16)]))
    
    # Family 4: Random
    for _ in range(10):
        initial_pop.append(sorted(rng.sample(range(1, MAX_VAL+1), MAX_N)))
    
    # Family 5: Log-spaced
    log_vals = np.logspace(0, np.log10(MAX_VAL), MAX_N)
    log_ints = sorted(set(int(v) for v in log_vals))
    while len(log_ints) < MAX_N:
        log_ints.append(log_ints[-1] + 1)
    if _is_valid(log_ints):
        initial_pop.append(log_ints[:MAX_N])
    
    # Filter and ensure we have enough valid individuals
    valid_pop = []
    for ind in initial_pop:
        if _is_valid(ind):
            valid_pop.append(ind)
    
    # Ensure minimum population size
    while len(valid_pop) < 20:
        valid_pop.append(sorted(rng.sample(range(1, MAX_VAL+1), MAX_N)))
    
    # Limit population size to avoid too many initial evaluations
    max_pop = min(len(valid_pop), 40)
    valid_pop = valid_pop[:max_pop]
    
    # Run genetic algorithm
    best_A, best_obj = _genetic_algorithm(valid_pop, time_budget, rng)
    
    # Final local refinement on the best
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0 and best_A:
        # Quick local search on best
        A_curr = sorted(best_A)
        curr_obj = _objective(A_curr)
        A_set = set(A_curr)
        
        for _ in range(int(min(500, remaining * 100))):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            idx = rng.randint(0, MAX_N-1)
            old_val = A_curr[idx]
            scale = max(1, old_val // 5)
            new_val = old_val + rng.randint(-scale, scale)
            new_val = max(1, min(MAX_VAL, new_val))
            if new_val in A_set:
                continue
            A_curr[idx] = new_val
            A_set.remove(old_val)
            A_set.add(new_val)
            new_obj = _objective(A_curr)
            if new_obj < curr_obj:
                curr_obj = new_obj
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best_A = list(A_curr)
            else:
                A_curr[idx] = old_val
                A_set.remove(new_val)
                A_set.add(old_val)
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
