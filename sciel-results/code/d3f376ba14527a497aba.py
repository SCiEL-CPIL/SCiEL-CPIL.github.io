"""HP lattice protein folding on the square lattice using pivot local search."""
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
    T_end = 0.05
    # Estimate iterations based on time limit, assuming ~0.1ms per iter for n=100
    # We will just loop until time limit is reached.
    start_time = time.time()
    
    # Pre-allocate for rotation calculations? No, just do it inline.
    # To speed up, we can pre-compute relative vectors? 
    # Let's just try a simple loop first.
    
    # We need to cool temperature. Since we don't know total iterations, 
    # we can use a fixed number of iterations or time-based.
    # Let's use a large number of iterations and break on time.
    max_iters = 100000 
    
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a pivot point (index of the residue before the one being rotated)
        # Pivot index k means we rotate the chain from k+1 to n-1 around chain[k]
        # k ranges from 0 to n-2
        k = rng.randint(0, n - 2)
        
        # The vector from k to k+1
        v_x = current_chain[k+1][0] - current_chain[k][0]
        v_y = current_chain[k+1][1] - current_chain[k][1]
        
        # Choose a rotation (90, 180, 270)
        rot_choice = rng.randint(1, 3)
        
        if rot_choice == 1:
            new_dx, new_dy = -v_y, v_x
        elif rot_choice == 2:
            new_dx, new_dy = -v_x, -v_y
        else:
            new_dx, new_dy = v_y, -v_x
            
        # Apply rotation to suffix
        px, py = current_chain[k]
        
        # Optimization: Check collision with prefix [0, k-1]
        # We can build a set of prefix positions. 
        # Building the set every time is O(k). 
        # For n=100, this is fine.
        prefix_occupied = set(current_chain[:k])
        
        valid = True
        new_chain = current_chain[:k+1]
        
        # We need to apply the rotation to all points j > k
        # new_pos[j] = (px, py) + rot(current_chain[j] - (px, py))
        
        # To avoid O(n) list creation if invalid early, we can just check first?
        # But we need the new_chain to compute energy if valid.
        # Let's just build it.
        
        for j in range(k + 1, n):
            ox = current_chain[j][0] - px
            oy = current_chain[j][1] - py
            
            if rot_choice == 1:
                nx, ny = -oy, ox
            elif rot_choice == 2:
                nx, ny = -ox, -oy
            else:
                nx, ny = oy, -ox
                
            new_pos = (px + nx, py + ny)
            
            if new_pos in prefix_occupied:
                valid = False
                break
                
            new_chain.append(new_pos)
            
        if not valid:
            # Also need to check if new_suffix self-intersects?
            # No, rotation is a rigid transformation, so self-intersection of suffix is preserved.
            # We only need to check against the fixed prefix.
            # Wait, is that true?
            # If the original suffix was self-avoiding, the rotated suffix is also self-avoiding.
            # The only new collisions can be with the prefix.
            # So the check above is sufficient.
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    # Allocate remaining time, but leave a safety margin
    remaining = _TOTAL_BUDGET - elapsed
    # We don't know how many sequences are left, but we can just use a chunk.
    # If this is the last one, it gets more. 
    # Simple strategy: use min(remaining, 15.0) seconds. 
    # If there are 10 sequences, 10*15=150s < 190s.
    time_limit = min(remaining, 15.0)
    
    if time_limit <= 1.0:
        # Not enough time for search, just return a valid chain
        # Greedy might be fast enough?
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Try multiple initial conformations
    # Allocate time to each. 
    # Let's do 3 restarts.
    num_restarts = 3
    time_per_restart = time_limit / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
