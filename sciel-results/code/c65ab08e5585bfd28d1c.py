"""HP lattice protein folding on the square lattice using pivot local search with greedy polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 190.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_pivot_move(sequence, chain, k, rot_choice):
    """
    Attempt a pivot move at index k with a specific rotation.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    px, py = chain[k]
    
    # Determine rotation
    v_x = chain[k+1][0] - px
    v_y = chain[k+1][1] - py
    
    if rot_choice == 1: # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 2: # 180 deg
        def rot(x, y): return (-x, -y)
    else: # 270 deg
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k])
    
    new_chain = list(chain[:k+1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
            
        new_chain.append(new_pos)
        
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.01
    start_time = time.time()
    
    # Estimate iterations. 
    # We will loop until time limit.
    # To ensure cooling, we need a max iteration count or time-based cooling.
    # Let's use time-based cooling: T = T_start * (T_end/T_start) ** (elapsed/total_est)
    # But we don't know total_est. 
    # Alternative: Fixed number of iterations, break on time.
    # For N=100, 100k iterations is ~10s. 
    # Let's aim for 50k-100k iterations.
    
    max_iters = 100000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        k = rng.randint(0, n - 2)
        rot_choice = rng.randint(1, 3)
        
        new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_rounds=50):
    """
    Greedy polish: repeatedly scan all possible pivot moves and accept any that improve energy.
    Stop when no move improves energy or max_rounds reached.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        for k in range(n - 1):
            for rot_choice in range(1, 4):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    # Restart scan from k=0 because chain changed
                    # To save time, we could just break and restart the outer loop
                    break
            if improved:
                break
                
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    # Allocate time. 
    # Reserve 5% for polish and overhead.
    sa_time = remaining * 0.9
    polish_time = remaining * 0.1
    
    if sa_time < 1.0:
        # Not enough time for SA, just greedy
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Dynamic number of restarts based on time
    # Estimate time per restart. 
    # Let's try 2 restarts for long sequences, 3 for short.
    num_restarts = 2 if n > 50 else 3
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine with SA
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    # Final Polish
    if time.time() - _GLOBAL_START < _TOTAL_BUDGET - 2.0: # Leave 2s margin
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
