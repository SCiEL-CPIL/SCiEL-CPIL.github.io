"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
DIR_NAMES = ['R', 'D', 'L', 'U']
VEC_TO_IDX = {(1,0):0, (0,-1):1, (-1,0):2, (0,1):3}

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _greedy_grow(sequence, rng, temperature=0.0):
    """
    Generate an initial valid conformation using greedy growth.
    Prioritizes creating H-H contacts, then maximizing free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: high gain, high free space, random tie-breaker
            score = gain * 10.0 + free * 0.5 + rng.random() * 1.0
            
            options.append((score, site))
            
        if not options:
            return None # Failed to grow
            
        # Sort by score descending and pick the best (or stochastic pick)
        options.sort(key=lambda x: -x[0])
        
        # Stochastic selection to allow diversity
        if temperature > 0:
            # Softmax-like selection
            max_score = options[0][0]
            weights = [math.exp((s - max_score) / temperature) for s, _ in options]
            total_w = sum(weights)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                chosen_site = options[0][1]
                for w, (_, site) in zip(weights, options):
                    cum += w
                    if r <= cum:
                        chosen_site = site
                        break
            else:
                chosen_site = options[0][1]
        else:
            chosen_site = options[0][1]
            
        chain.append(chosen_site)
        occupied[chosen_site] = i
        
    return chain

def _regrow_suffix(sequence, chain, start_idx, rng, temperature=1.0):
    """
    Regrow the chain from start_idx to the end.
    Returns the new chain or None if it fails.
    """
    n = len(sequence)
    if start_idx >= n:
        return list(chain)
    
    # Keep prefix
    new_chain = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(new_chain)}
    
    # Regrow
    for i in range(start_idx, n):
        last = new_chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Heuristic score for regrowth
            score = gain * 5.0 + free * 0.2 + rng.random() * 2.0
            options.append((score, site))
            
        if not options:
            return None
            
        # Stochastic choice based on temperature
        # Lower temperature -> more greedy
        if temperature < 0.1:
            options.sort(key=lambda x: -x[0])
            chosen_site = options[0][1]
        else:
            # Boltzmann selection
            max_score = max(s for s, _ in options)
            weights = [math.exp((s - max_score) / temperature) for s, _ in options]
            total_w = sum(weights)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                chosen_site = options[0][1]
                for w, (_, site) in zip(weights, options):
                    cum += w
                    if r <= cum:
                        chosen_site = site
                        break
            else:
                chosen_site = options[0][1]
                
        new_chain.append(chosen_site)
        occupied[chosen_site] = i
        
    return new_chain

def _sa_loop(sequence, initial_chain, rng, time_limit):
    """
    Simulated Annealing loop using regrow moves.
    """
    n = len(sequence)
    current_chain = initial_chain
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Initial temperature
    # Energy scale is roughly -N/2 to 0. 
    # Delta E is usually small (1-3).
    start_temp = 5.0
    end_temp = 0.1
    cool_rate = 0.999
    
    temp = start_temp
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        if temp < end_temp:
            break
            
        # Pick a random cut point for regrow
        # Avoid cutting too early (too expensive) or too late (too small)
        # Cut point k: regrow k..N-1. 
        # We want to change a significant portion.
        # Random k between 5 and N-5
        min_cut = min(5, n - 1)
        max_cut = max(min_cut + 1, n - 2)
        
        if n < 10:
            k = rng.randint(0, n - 1)
        else:
            k = rng.randint(min_cut, max_cut)
            
        # Regrow
        new_chain = _regrow_suffix(sequence, current_chain, k, rng, temp)
        
        if new_chain is None:
            temp *= cool_rate
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 0.01))):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temp *= cool_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Multi-Start Regrow Simulated Annealing.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    best_chain = None
    best_energy = 0
    
    # Time allocation strategy:
    # 1. Generate a few diverse initial solutions using Greedy Growth with different temps/seeds.
    # 2. Run SA on each for a portion of the time.
    
    # We have ~170s total.
    # Let's do 5 independent runs.
    num_runs = 5
    time_per_run = 30.0 # 5 * 30 = 150s, leaving 20s buffer
    
    # For very short sequences, we can do more runs
    if n < 20:
        num_runs = 10
        time_per_run = 15.0
    elif n > 80:
        num_runs = 3
        time_per_run = 50.0
        
    for run_id in range(num_runs):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate initial conformation
        # Use different greedy temperatures to get diverse starts
        greedy_temp = 0.5 + run_id * 0.5
        
        init_chain = _greedy_grow(sequence, rng, temperature=greedy_temp)
        
        if init_chain is None:
            # Fallback: try straight line or simple random
            # Try greedy with temp 0
            init_chain = _greedy_grow(sequence, rng, temperature=0.0)
            if init_chain is None:
                continue
                
        # Run SA
        sa_chain, sa_energy = _sa_loop(sequence, init_chain, rng, time_per_run)
        
        if best_chain is None or sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Final polish on the best solution found
    if best_chain is not None:
        # Short, cool SA pass to clean up
        final_chain, final_energy = _sa_loop(sequence, best_chain, rng, 10.0)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy

    # Fallback if everything failed (should not happen)
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
