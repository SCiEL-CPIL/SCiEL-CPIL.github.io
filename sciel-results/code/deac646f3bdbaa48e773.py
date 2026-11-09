<<<<<<< SEARCH
def _simulated_annealing(order, P, start_time, time_limit):
    """Simulated annealing with swap and or-opt moves."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(order, P)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.99
    min_temperature = 0.1
    max_steps = 50000
    
    rng = random.Random(42)
    
    step = 0
    while (time.time() - start_time) < time_limit and T > min_temperature and step < max_steps:
        step += 1
        
        # Generate a neighbor move
        move_type = rng.choice(['swap', 'or_opt'])
        
        if move_type == 'swap' and n_jobs >= 2:
            i, j = rng.sample(range(n_jobs), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 'or_opt' and n_jobs >= 3:
            # Or-opt: move a block of 1, 2, or 3 jobs to a new position
            block_len = rng.choice([1, 2, 3])
            if n_jobs < block_len + 1:
                block_len = max(1, n_jobs - 1)
            
            i = rng.randint(0, n_jobs - block_len)
            block = current_order[i:i+block_len]
            
            # Remove block
            new_order = current_order[:i] + current_order[i+block_len:]
            
            # Insert block at random position (excluding original position to ensure a move)
            j = rng.randint(0, len(new_order))
            
            new_order = new_order[:j] + block + new_order[j:]
            
            if new_order == current_order:
                continue
        else:
            continue
        
        new_makespan = _makespan(new_order, P)
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Accept improvement
            current_order = new_order
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_order = current_order[:]
                best_makespan = current_makespan
        else:
            # Accept worse solution with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                current_order = new_order
                current_makespan = new_makespan
        
        T *= cooling_rate
    
    return best_order, best_makespan
=======
def _ils_search(order, P, start_time, time_limit):
    """Iterated Local Search with Or-opt and Reversal perturbations."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(order, P)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    rng = random.Random(42)
    
    T = 5.0
    cooling_rate = 0.995
    min_temperature = 0.5
    
    while (time.time() - start_time) < start_time + time_limit:
        # Intensify: Local Search
        # Use fewer passes to allow more iterations
        current_order, current_makespan = _insertion_search(
            current_order, P, max_passes=5, time_limit=None, start_time=None
        )
        current_order, current_makespan = _or_opt_search(
            current_order, P, max_passes=5, time_limit=None, start_time=None
        )
        
        if current_makespan < best_makespan:
            best_order = current_order[:]
            best_makespan = current_makespan
            
        if (time.time() - start_time) >= start_time + time_limit:
            break
            
        # Diversify: Perturbation
        move_type = rng.choice(['or_opt', 'reversal'])
        
        if move_type == 'or_opt':
            block_len = rng.randint(2, min(4, n_jobs))
            if n_jobs > block_len:
                i = rng.randint(0, n_jobs - block_len)
                block = current_order[i:i+block_len]
                remaining = current_order[:i] + current_order[i+block_len:]
                j = rng.randint(0, len(remaining))
                new_order = remaining[:j] + block + remaining[j:]
            else:
                continue
        else: # reversal
            if n_jobs >= 3:
                i, j = rng.sample(range(n_jobs), 2)
                if i > j: i, j = j, i
                new_order = current_order[:]
                new_order[i:j+1] = reversed(new_order[i:j+1])
            else:
                continue
                
        if new_order == current_order:
            continue
            
        new_makespan = _makespan(new_order, P)
        delta = new_makespan - current_makespan
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_order = new_order
            current_makespan = new_makespan
            
        T *= cooling_rate
        if T < min_temperature:
            T = min_temperature

    return best_order, best_makespan
>>>>>>> REPLACE

Now update `order_jobs` to use `_ils_search`.

