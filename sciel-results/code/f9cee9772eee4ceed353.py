"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, pt):
    """Compute makespan for a given permutation using vectorized numpy."""
    n = len(perm)
    if n == 0:
        return 0.0
    # pt is (jobs, machines), perm is array of job indices
    # completion[j, m] = completion[j, m-1] + pt[j, m]
    # completion[j, m] = max(completion[j, m], completion[j-1, m]) + pt[j, m]  ... wait that's not right
    # Actually for flow shop:
    # C(j, m) = C(j, m-1) + p(j, m)  for m=0
    # C(j, m) = max(C(j, m-1), C(j-1, m)) + p(j, m)  for m>0
    # Makespan = C(n-1, M-1)
    
    # Let's do it properly with numpy
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Get processing times in order of permutation
    p = pt[perm]  # shape (n_jobs, n_machines)
    
    # Build completion matrix
    # C[i, j] = max(C[i, j-1], C[i-1, j]) + p[i, j]
    # C[i, 0] = C[i-1, 0] + p[i, 0]
    # C[0, j] = C[0, j-1] + p[0, j]
    
    # We can compute this iteratively, but let's try to be efficient
    # For small n_jobs this is fine
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    C[0, 0] = p[0, 0]
    for j in range(1, n_machines):
        C[0, j] = C[0, j-1] + p[0, j]
    for i in range(1, n_jobs):
        C[i, 0] = C[i-1, 0] + p[i, 0]
        for j in range(1, n_machines):
            C[i, j] = max(C[i, j-1], C[i-1, j]) + p[i, j]
    
    return C[-1, -1]


def _neh_order(pt, random_state=None):
    """NEH heuristic for flow shop."""
    if random_state is None:
        random_state = random.Random()
    
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build permutation by inserting each job in best position
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(np.array(candidate), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _johnsons_rule(pt):
    """Johnson's rule for 2-machine flow shop. For >2 machines, use approximation."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs == 0:
        return []
    
    if n_machines == 2:
        # Standard Johnson's rule
        a = pt[:, 0]
        b = pt[:, 1]
        min_ab = np.minimum(a, b)
        
        # Jobs where a <= b go in first set, others in second
        set1 = [i for i in range(n_jobs) if a[i] <= b[i]]
        set2 = [i for i in range(n_jobs) if a[i] > b[i]]
        
        set1.sort(key=lambda i: a[i])
        set2.sort(key=lambda i: b[i], reverse=True)
        
        return set1 + set2
    else:
        # Approximation: use NEH
        return _neh_order(pt)


def _swap_mutation(perm, random_state):
    """Swap two random positions."""
    perm = list(perm)
    if len(perm) < 2:
        return perm
    i, j = random_state.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def _insert_mutation(perm, random_state):
    """Remove a job and insert it elsewhere."""
    perm = list(perm)
    if len(perm) < 2:
        return perm
    i = random_state.randint(0, len(perm) - 1)
    job = perm.pop(i)
    j = random_state.randint(0, len(perm))
    perm.insert(j, job)
    return perm


def _reversal_mutation(perm, random_state):
    """Reverse a random subsequence."""
    perm = list(perm)
    if len(perm) < 2:
        return perm
    i, j = random_state.sample(range(len(perm)), 2)
    if i > j:
        i, j = j, i
    perm[i:j+1] = perm[i:j+1][::-1]
    return perm


def _order_crossover(parent1, parent2, random_state):
    """Order crossover (OX1)."""
    n = len(parent1)
    if n < 3:
        # For small permutations, just return a random permutation
        perm = list(parent1)
        random_state.shuffle(perm)
        return perm
    
    i, j = random_state.sample(range(n), 2)
    if i > j:
        i, j = j, i
    
    child = [None] * n
    child[i:j+1] = parent1[i:j+1]
    
    # Fill remaining positions with jobs from parent2 in order
    fill = [x for x in parent2 if x not in parent1[i:j+1]]
    pos = 0
    for k in range(n):
        if child[k] is None:
            child[k] = fill[pos]
            pos += 1
    
    return child


def _local_search(perm, pt, max_iterations=50, time_limit=None, start_time=None):
    """Local search with swap and insert moves."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 5.0
    
    perm = list(perm)
    n = len(perm)
    if n == 0:
        return perm
    
    current_ms = _makespan(np.array(perm), pt)
    best_perm = perm[:]
    best_ms = current_ms
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all swap moves
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                perm[i], perm[j] = perm[j], perm[i]
                new_ms = _makespan(np.array(perm), pt)
                if new_ms < current_ms - 1e-9:
                    current_ms = new_ms
                    improved = True
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_perm = perm[:]
                else:
                    perm[i], perm[j] = perm[j], perm[i]
        
        if not improved:
            # Try insert moves
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                job = perm[i]
                perm.pop(i)
                for j in range(len(perm)):
                    perm.insert(j, job)
                    new_ms = _makespan(np.array(perm), pt)
                    if new_ms < current_ms - 1e-9:
                        current_ms = new_ms
                        improved = True
                        if new_ms < best_ms:
                            best_ms = new_ms
                            best_perm = perm[:]
                        break
                    else:
                        perm.pop(j)
                        if j < i:
                            perm.insert(i, job)
                        else:
                            perm.insert(i, job)
                # Restore if not improved
                if not improved:
                    perm.insert(i, job)
        
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    """Main entry point: GA + local search for flow shop scheduling."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    random_state = random.Random(42)
    
    # Initialize population with diverse heuristics
    pop = []
    
    # NEH
    neh = _neh_order(pt, random_state)
    pop.append(neh)
    
    # Johnson's rule (or NEH for >2 machines)
    johnson = _johnsons_rule(pt)
    if johnson != neh:
        pop.append(johnson)
    
    # Random permutations
    for _ in range(3):
        perm = list(range(n_jobs))
        random_state.shuffle(perm)
        pop.append(perm)
    
    # Ensure unique
    seen = set()
    unique_pop = []
    for p in pop:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_pop.append(p)
    pop = unique_pop
    
    # Evaluate initial population
    pop_ms = [_makespan(np.array(p), pt) for p in pop]
    
    best_perm = pop[np.argmin(pop_ms)]
    best_ms = min(pop_ms)
    
    # GA parameters
    pop_size = max(10, min(30, n_jobs * 2))
    while len(pop) < pop_size:
        # Add more random or mutated individuals
        base = random_state.choice(pop)
        if random_state.random() < 0.5:
            child = _swap_mutation(base, random_state)
        elif random_state.random() < 0.5:
            child = _insert_mutation(base, random_state)
        else:
            child = _reversal_mutation(base, random_state)
        pop.append(child)
    
    # Re-evaluate
    pop_ms = [_makespan(np.array(p), pt) for p in pop]
    best_idx = np.argmin(pop_ms)
    best_perm = pop[best_idx]
    best_ms = pop_ms[best_idx]
    
    # GA loop
    generation = 0
    while time.time() - start_time < deadline:
        generation += 1
        
        # Selection: tournament
        def tournament_select():
            idxs = random_state.sample(range(len(pop)), min(3, len(pop)))
            best = idxs[0]
            for idx in idxs[1:]:
                if pop_ms[idx] < pop_ms[best]:
                    best = idx
            return best
        
        new_pop = []
        
        # Elitism: keep best individual
        new_pop.append(best_perm[:])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() - start_time > deadline:
                break
            
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            
            if random_state.random() < 0.7:
                child = _order_crossover(pop[p1_idx], pop[p2_idx], random_state)
            else:
                child = _order_crossover(pop[p1_idx], pop[p2_idx], random_state)
            
            # Mutation
            r = random_state.random()
            if r < 0.4:
                child = _swap_mutation(child, random_state)
            elif r < 0.7:
                child = _insert_mutation(child, random_state)
            else:
                child = _reversal_mutation(child, random_state)
            
            new_pop.append(child)
        
        # Evaluate new population
        new_pop_ms = [_makespan(np.array(p), pt) for p in new_pop]
        
        # Combine and select
        all_pop = new_pop
        all_ms = new_pop_ms
        
        # Update best
        best_idx = np.argmin(all_ms)
        if all_ms[best_idx] < best_ms:
            best_ms = all_ms[best_idx]
            best_perm = all_pop[best_idx][:]
        
        pop = all_pop
        pop_ms = all_ms
        
        # Apply local search to top few individuals
        top_k = min(3, len(pop))
        sorted_indices = np.argsort(pop_ms)[:top_k]
        
        for idx in sorted_indices:
            if time.time() - start_time > deadline:
                break
            local_start = time.time()
            remaining = deadline - local_start
            if remaining < 0.1:
                break
            improved = _local_search(pop[idx], pt, max_iterations=30, time_limit=min(remaining, 3.0), start_time=local_start)
            improved_ms = _makespan(np.array(improved), pt)
            if improved_ms < pop_ms[idx]:
                pop[idx] = improved
                pop_ms[idx] = improved_ms
            if improved_ms < best_ms:
                best_ms = improved_ms
                best_perm = improved[:]
    
    # Final local search on best
    if time.time() - start_time < deadline - 1.0:
        remaining = deadline - 1.0 - (time.time() - start_time)
        if remaining > 0.1:
            final = _local_search(best_perm, pt, max_iterations=100, time_limit=remaining, start_time=time.time())
            final_ms = _makespan(np.array(final), pt)
            if final_ms < best_ms:
                best_perm = final
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
