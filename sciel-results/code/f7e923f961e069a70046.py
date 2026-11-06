"""HP lattice protein folding using a Genetic Algorithm with Local Search."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    contacts = 0
    for i in h_positions:
        pt = tuple(chain[i])
        for q in _neighbours(pt):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2

def _is_valid_chain(chain):
    """Check if chain is self-avoiding and connected."""
    n = len(chain)
    if n == 0:
        return True
    occupied = set()
    for p in chain:
        tp = tuple(p)
        if tp in occupied:
            return False
        occupied.add(tp)
    for i in range(1, n):
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            return False
    return True

def _generate_random_walk(n, rng):
    """Generate a random self-avoiding walk."""
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for _ in range(1, n):
        options = []
        for site in _neighbours(tuple(chain[-1])):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = len(chain)
        chain.append(site)
    return chain

def _generate_compact_walk(n, rng):
    """Generate a compact spiral-like walk."""
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_len = 1
    
    for _ in range(4 * n): # Max steps
        if len(chain) >= n:
            break
        dx, dy = directions[dir_idx]
        for _ in range(step_len):
            if len(chain) >= n:
                break
            new_pt = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pt not in occupied:
                occupied[new_pt] = len(chain)
                chain.append(new_pt)
            else:
                # Try to find any free neighbor to continue
                found = False
                for d in directions:
                    alt = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                    if alt not in occupied:
                        occupied[alt] = len(chain)
                        chain.append(alt)
                        dir_idx = directions.index(d)
                        found = True
                        break
                if not found:
                    break
        dir_idx = (dir_idx + 1) % 4
        if dir_idx % 2 == 0:
            step_len += 1
            
    # Fill remaining with random if spiral failed
    while len(chain) < n:
        options = []
        for site in _neighbours(tuple(chain[-1])):
            if site not in occupied:
                options.append(site)
        if not options:
            break
        site = rng.choice(options)
        occupied[site] = len(chain)
        chain.append(site)
        
    return chain[:n]

def _reverse_subchain(chain, i, j):
    """Reverse subchain i..j and check validity."""
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    return new_chain if _is_valid_chain(new_chain) else None

def _mutate(chain, rng):
    """Perform a random reversal mutation."""
    n = len(chain)
    if n < 3:
        return chain
    i = rng.randint(0, n - 2)
    j = rng.randint(i + 1, n - 1)
    new_chain = _reverse_subchain(chain, i, j)
    if new_chain is None:
        # Try a few more times
        for _ in range(5):
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _reverse_subchain(chain, i, j)
            if new_chain:
                break
    return new_chain if new_chain else chain

def _crossover(p1, p2, rng):
    """1-point crossover. Returns valid child or None."""
    n = len(p1)
    if n < 4:
        return None
    cut = rng.randint(1, n - 2)
    # Child: head of p1, tail of p2
    # We need to shift the tail of p2 so it connects to head of p1
    # Actually, simple splicing often breaks connectivity.
    # A better crossover for SAWs is to take a segment from one and a segment from another
    # and try to join them.
    
    # Let's try a "Splice" where we take head of p1 and tail of p2.
    # The connection point is p1[cut-1] and p2[cut].
    # They must be adjacent. If not, we can try to rotate/shift the tail of p2.
    # This is complex. Let's stick to a simpler "Segment Swap" or just rely on Mutation
    # if Crossover is too risky.
    
    # Alternative: Crossover by taking head of p1 up to cut, and tail of p2 from cut.
    # We need to move the tail of p2 so that p2[cut] is adjacent to p1[cut-1].
    # This requires a translation.
    
    head = [list(p) for p in p1[:cut]]
    tail = [list(p) for p in p2[cut:]]
    
    if not head or not tail:
        return None
        
    anchor = tuple(head[-1])
    start_tail = tuple(tail[0])
    
    # Check if start_tail is already adjacent to anchor
    # If not, try to shift the entire tail so that start_tail moves to a neighbor of anchor
    # that is free.
    
    # For simplicity and robustness in a "new" GA, let's use a "Recombination" that
    # is less likely to fail: take the first half of one and second half of the other,
    # but only if they can be connected by a single step (which is rare).
    
    # Let's use a different strategy: **Double-Edge Crossover** or just **Mutation-Only GA**
    # which is essentially a population-based Hill Climber.
    # Given the time constraints, a robust Mutation-Only GA (Memetic) is safer.
    
    return None # We will use mutation only in this implementation for reliability

def _local_search(chain, sequence, rng, steps=100):
    """Hill climbing using reversals."""
    current_energy = _compute_energy(sequence, chain)
    best_chain = [list(p) for p in chain]
    best_energy = current_energy
    
    for _ in range(steps):
        n = len(chain)
        if n < 3:
            break
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        new_chain = _reverse_subchain(chain, i, j)
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        if new_energy < current_energy:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in chain]
        # Accept worse moves occasionally to escape local minima (SA-like)
        elif rng.random() < 0.1:
            chain = new_chain
            current_energy = new_energy
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Genetic Algorithm with Local Search."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    total_budget = 10.0 # seconds
    start_time = time.time()
    
    # Population size
    pop_size = 20
    population = []
    
    # Initialize population
    for i in range(pop_size):
        if i % 2 == 0:
            chain = _generate_compact_walk(n, random.Random(100+i))
        else:
            chain = _generate_random_walk(n, random.Random(200+i))
            
        if chain and len(chain) == n:
            population.append(chain)
            
    # If population is small, fill with random
    while len(population) < pop_size:
        chain = _generate_random_walk(n, random.Random(300+len(population)))
        if chain:
            population.append(chain)
        else:
            break
            
    if not population:
        return [[i, 0] for i in range(n)]
        
    best_chain = population[0]
    best_energy = _compute_energy(sequence, best_chain)
    
    generation = 0
    rng = random.Random(42)
    
    while (time.time() - start_time) < total_budget * 0.9:
        generation += 1
        new_population = []
        
        # Elitism: Keep the best
        population.sort(key=lambda c: _compute_energy(sequence, c))
        new_population.append(population[0])
        
        # Create offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > total_budget:
                break
                
            # Select parent
            # Tournament selection
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            
            # Try crossover (currently returns None, so fallback to mutation)
            child = _crossover(p1, p2, rng)
            if child is None:
                # Mutate the better parent
                parent = p1 if _compute_energy(sequence, p1) <= _compute_energy(sequence, p2) else p2
                child = _mutate(parent, rng)
                
            if child is None:
                child = [list(p) for p in parent]
                
            # Local Search on child
            improved_chain, improved_energy = _local_search(child, sequence, rng, steps=50)
            new_population.append(improved_chain)
            
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
        
        population = new_population
        
    return best_chain
# EVOLVE-BLOCK-END
