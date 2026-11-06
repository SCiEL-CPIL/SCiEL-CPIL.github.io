import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 256
    
    # Build conflict graph: adj[i] = set of j where (i*j) % (i+j) == 0
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
    
    # Precompute degrees
    degree = [len(adj[i]) for i in range(N + 1)]
    
    # Greedy initialization: min-degree with random tie-breaking
    def greedy_init():
        available = set(range(1, N + 1))
        iset = set()
        while available:
            if time.time() > deadline:
                break
            # Find min-degree node in available
            min_deg = float('inf')
            candidates = []
            for node in available:
                deg = len(adj[node] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [node]
                elif deg == min_deg:
                    candidates.append(node)
            selected = random.choice(candidates)
            iset.add(selected)
            available.difference_update(adj[selected])
            available.discard(selected)
        return iset
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Simulated annealing
    def run_sa(initial_set, time_budget):
        sa_deadline = start_time + time_budget
        current = set(initial_set)
        current_size = len(current)
        best_local = set(current)
        best_local_size = current_size
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        swap_prob = 0.5
        
        while time.time() < sa_deadline and T > min_T:
            # Choose a move
            move_type = random.random()
            if move_type < 0.3:
                # Try to add a random node
                # Find nodes not in current that don't conflict
                not_in = [i for i in range(1, N + 1) if i not in current]
                if not not_in:
                    T *= cooling
                    continue
                candidate = random.choice(not_in)
                if adj[candidate] & current:
                    # Conflict, skip
                    continue
                new_set = set(current)
                new_set.add(candidate)
                delta = 1
                accept = True
            elif move_type < 0.8:
                # Swap: remove one, add another
                if not current:
                    T *= cooling
                    continue
                to_remove = random.choice(list(current))
                not_in = [i for i in range(1, N + 1) if i not in current]
                if not not_in:
                    T *= cooling
                    continue
                to_add = random.choice(not_in)
                if to_add == to_remove:
                    T *= cooling
                    continue
                # Check if to_add conflicts with current minus to_remove
                remaining = current - {to_remove}
                if adj[to_add] & remaining:
                    continue
                new_set = set(current)
                new_set.discard(to_remove)
                new_set.add(to_add)
                delta = 0
                # Accept if same size with some probability, or use energy
                # For same-size, use a small acceptance probability
                accept = random.random() < 0.3
            else:
                # Remove a random node (to allow exploration)
                if not current:
                    T *= cooling
                    continue
                to_remove = random.choice(list(current))
                new_set = set(current)
                new_set.discard(to_remove)
                delta = -1
                # Only accept removals at high T
                accept = random.random() < 0.1 and T > 1.0
            
            if not accept:
                T *= cooling
                continue
            
            # Evaluate
            new_size = len(new_set)
            if delta > 0:
                # Always accept improvements
                current = new_set
                current_size = new_size
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local = set(current)
            elif delta == 0:
                # Same size, accept with some probability (already checked)
                if accept:
                    current = new_set
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = set(current)
            else:
                # Removal, accept with probability based on T
                if accept:
                    current = new_set
                    current_size = new_size
            
            T *= cooling
        
        return best_local, best_local_size
    
    # Run multiple SA instances with different seeds
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        # Create a perturbed initial set
        if num_restarts == 1:
            init_set = best_set
        else:
            # Perturb: remove some random elements, re-greedy
            perturbed = set(best_set)
            num_remove = random.randint(5, 20)
            for _ in range(num_remove):
                if perturbed:
                    perturbed.discard(random.choice(list(perturbed)))
            # Fill with greedy
            available = set(range(1, N + 1)) - perturbed
            while available:
                if time.time() > deadline:
                    break
                min_deg = float('inf')
                candidates = []
                for node in available:
                    deg = len(adj[node] & (available | perturbed))
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [node]
                    elif deg == min_deg:
                        candidates.append(node)
                selected = random.choice(candidates)
                perturbed.add(selected)
                available.difference_update(adj[selected])
                available.discard(selected)
            init_set = perturbed
        
        # Run SA with remaining time budget
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        sa_time = min(remaining * 0.8, 15)  # Cap each SA run
        local_best, local_size = run_sa(init_set, sa_time)
        
        if local_size > best_size:
            best_size = local_size
            best_set = local_best
        
        # Early exit if we've found a very large set
        if best_size > 195:
            # Try one more focused SA
            remaining = deadline - 5 - time.time()
            if remaining > 2:
                local_best, local_size = run_sa(best_set, remaining)
                if local_size > best_size:
                    best_size = local_size
                    best_set = local_best
            break
    
    return list(best_set)
