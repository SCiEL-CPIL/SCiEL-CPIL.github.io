"""HP lattice protein folding on the square lattice using simulated annealing with pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP contact energy: -1 per H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_chain(sequence, dimension):
    """Create an initial self-avoiding walk using a spiral-like pattern."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Try to grow in a spiral pattern
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_in_dir = 1
    next_dir_change = 1
    
    for i in range(1, n):
        placed = False
        # Try to continue in current direction, then try turning
        for attempt in range(4):
            dx, dy = directions[dir_idx]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                step_count += 1
                if step_count == next_dir_change:
                    dir_idx = (dir_idx + 1) % 4
                    step_count = 0
                    steps_in_dir += 1
                    next_dir_change = steps_in_dir
                break
            else:
                dir_idx = (dir_idx + 1) % 4
                step_count = 0
                steps_in_dir += 1
                next_dir_change = steps_in_dir
        
        if not placed:
            # Fallback: just extend in a straight line
            chain.append((chain[-1][0] + i, 0))
    
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n <= 2:
        return None
    
    # Choose pivot point (not the first or last)
    pivot = rng.randint(1, n - 2)
    
    # Choose rotation angle: 90 degrees clockwise or counterclockwise
    angle = rng.choice([1, -1])
    
    # Extract suffix to rotate
    suffix = chain[pivot + 1:]
    
    # Apply rotation to each point in the suffix relative to the pivot
    rotated = []
    for pt in suffix:
        dx = pt[0] - chain[pivot][0]
        dy = pt[1] - chain[pivot][1]
        if angle == 1:
            # Clockwise: (x, y) -> (y, -x)
            new_dx, new_dy = dy, -dx
        else:
            # Counterclockwise: (x, y) -> (-y, x)
            new_dx, new_dy = -dy, dx
        rotated.append((chain[pivot][0] + new_dx, chain[pivot][1] + new_dy))
    
    # Check for self-avoidance: the rotated suffix must not collide with the prefix
    prefix = chain[:pivot + 1]
    prefix_set = set(prefix)
    
    for pt in rotated:
        if pt in prefix_set:
            return None
    
    # Also check that the rotated suffix is self-avoiding (it should be since rotation preserves distances)
    # But we need to check against itself
    rotated_set = set(rotated)
    if len(rotated_set) != len(rotated):
        return None
    
    return prefix + rotated


def _compute_energy(sequence, chain):
    """Compute energy efficiently using a contact map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, rng, max_iterations=1000):
    """Perform a local search using single-residue moves to find a local optimum."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iterations):
        # Try moving a random residue to a neighbouring empty site
        i = rng.randint(1, n - 2)  # Don't move first or last
        moved = False
        for q in _neighbours(chain[i]):
            if q == chain[i - 1] or q == chain[i + 1]:
                continue
            # Check if q is occupied
            occupied = set(chain)
            if q in occupied:
                continue
            
            # Try the move
            new_chain = list(chain)
            new_chain[i] = q
            
            # Check self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Compute new energy
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                moved = True
                break
        
        if not moved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using simulated annealing with pivot moves and local search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Set a deadline: 180s total for all sequences, so use most of it
    start_time = time.time()
    deadline = start_time + 170  # 170s safety margin
    
    # Initial chain
    chain = _initial_chain(sequence, dimension)
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing
    rng = random.Random(42 + len(sequence))
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    iterations = 0
    while T > T_min:
        # Check time
        if time.time() > deadline:
            break
        
        # Perform a pivot move
        new_chain = _pivot_move(chain, rng)
        
        if new_chain is not None:
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        # Cool down
        T *= cooling_rate
        iterations += 1
        
        # Periodically run local search
        if iterations % 500 == 0 and time.time() < deadline - 5:
            local_chain, local_energy = _local_search(sequence, chain, rng, max_iterations=200)
            if local_energy < best_energy:
                best_energy = local_energy
                best_chain = local_chain
            chain = local_chain
            current_energy = local_energy
    
    # Final local search
    if time.time() < deadline - 1:
        local_chain, local_energy = _local_search(sequence, best_chain, rng, max_iterations=500)
        if local_energy < best_energy:
            best_energy = local_energy
            best_chain = local_chain
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
