"""HP lattice protein folding using simulated annealing with pivot and end moves."""
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
    """Compute energy as negative count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _initial_spiral(sequence, dimension):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Start with a simple spiral pattern
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    dx, dy = 1, 0  # start moving right
    
    for i in range(1, n):
        # Try to continue in current direction
        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
        if new_pos not in occupied:
            chain.append(new_pos)
            occupied[new_pos] = i
        else:
            # Turn right (clockwise)
            dx, dy = -dy, dx
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
            else:
                # Try turning left
                dx, dy = dy, -dx
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                else:
                    # Backtrack - just place in a straight line as fallback
                    break
    
    # If spiral failed, use straight line
    if len(chain) < n:
        chain = [(i, 0) for i in range(n)]
    
    return chain


def _pivot_move(chain, pivot_idx, sequence, rng):
    """
    Perform a pivot move at the given pivot index.
    Returns (new_chain, is_valid).
    """
    n = len(chain)
    if n < 3:
        return None, False
    
    static_part = chain[:pivot_idx + 1]
    moving_part = chain[pivot_idx + 1:]
    
    pivot_point = chain[pivot_idx]
    
    # Try all 3 non-identity rotations
    rotations = [
        lambda x, y: (y, -x),    # 90 degrees clockwise
        lambda x, y: (-x, -y),   # 180 degrees
        lambda x, y: (-y, x),    # 90 degrees counter-clockwise
    ]
    rot_func = rng.choice(rotations)
    
    static_occupied = set(static_part)
    new_moving_part = []
    new_moving_occupied = set()
    is_valid = True
    
    for pos in moving_part:
        x, y = pos
        dx, dy = x - pivot_point[0], y - pivot_point[1]
        nx, ny = rot_func(dx, dy)
        new_pos = (pivot_point[0] + nx, pivot_point[1] + ny)
        
        if new_pos in static_occupied:
            is_valid = False
            break
        if new_pos in new_moving_occupied:
            is_valid = False
            break
            
        new_moving_part.append(new_pos)
        new_moving_occupied.add(new_pos)
    
    if not is_valid:
        return None, False
    
    # Verify connectivity at the pivot
    last_static = static_part[-1]
    first_new_moving = new_moving_part[0]
    dist = abs(last_static[0] - first_new_moving[0]) + abs(last_static[1] - first_new_moving[1])
    if dist != 1:
        return None, False
    
    new_chain = static_part + new_moving_part
    return new_chain, True


def _end_move(chain, sequence, rng):
    """
    Attempt an end move.
    Returns (new_chain, is_valid).
    """
    n = len(chain)
    if n < 3:
        return None, False
    
    L = rng.choice([2, 3])
    if L >= n:
        return None, False
        
    static_part = chain[:-L]
    moving_part = chain[-L:]
    
    pivot_point = static_part[-1]
    
    rotations = [
        lambda x, y: (y, -x),
        lambda x, y: (-x, -y),
        lambda x, y: (-y, x)
    ]
    rot_func = rng.choice(rotations)
    
    static_occupied = set(static_part)
    new_moving_part = []
    new_moving_occupied = set()
    is_valid = True
    
    for pos in moving_part:
        x, y = pos
        dx, dy = x - pivot_point[0], y - pivot_point[1]
        nx, ny = rot_func(dx, dy)
        new_pos = (pivot_point[0] + nx, pivot_point[1] + ny)
        
        if new_pos in static_occupied:
            is_valid = False
            break
        if new_pos in new_moving_occupied:
            is_valid = False
            break
            
        new_moving_part.append(new_pos)
        new_moving_occupied.add(new_pos)
    
    if not is_valid:
        return None, False
        
    last_static = static_part[-1]
    first_new_moving = new_moving_part[0]
    dist = abs(last_static[0] - first_new_moving[0]) + abs(last_static[1] - first_new_moving[1])
    if dist != 1:
        return None, False
        
    new_chain = static_part + new_moving_part
    return new_chain, True


def _simulated_annealing(sequence, dimension, rng, time_limit):
    """
    Perform simulated annealing with pivot and end moves.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return _initial_spiral(sequence, dimension)
    
    # Start from a diverse initial solution
    start_chain = _initial_spiral(sequence, dimension)
    current_chain = start_chain
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = current_chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    initial_temp = 1.0
    cooling_rate = 0.999
    
    temp = initial_temp
    no_improve_counter = 0
    max_no_improve = 2000
    
    while temp > 0.01 and time.time() - start_time < time_limit:
        # Choose move type: 70% pivot, 30% end
        if rng.random() < 0.7:
            pivot_idx = rng.randint(0, n - 2)
            new_chain, is_valid = _pivot_move(current_chain, pivot_idx, sequence, rng)
        else:
            new_chain, is_valid = _end_move(current_chain, sequence, rng)
        
        if not is_valid or new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / temp)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = current_chain[:]
                best_energy = current_energy
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        temp *= cooling_rate
        
        # If stuck for too long, restart with a new initial solution
        if no_improve_counter > max_no_improve:
            no_improve_counter = 0
            temp = max(temp, 0.5)
            # Try a fresh initial
            fresh = _initial_spiral(sequence, dimension)
            fresh_energy = _compute_energy(sequence, fresh)
            if fresh_energy < best_energy:
                best_chain = fresh[:]
                best_energy = fresh_energy
            # With some probability, jump to the fresh solution
            if rng.random() < 0.3:
                current_chain = fresh[:]
                current_energy = fresh_energy
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using simulated annealing with pivot moves."""
    n = len(sequence)
    
    if n <= 1:
        return [[0, 0]]
    
    # Allocate time budget with safety margin (210s total, use ~200s)
    start = time.time()
    time_budget = 190.0  # seconds
    
    # Determine number of restarts based on sequence length
    if n <= 30:
        num_restarts = 12
    elif n <= 50:
        num_restarts = 8
    elif n <= 70:
        num_restarts = 5
    else:
        num_restarts = 3
    
    time_per_restart = time_budget / num_restarts
    
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        remaining = time_budget - (time.time() - start)
        if remaining < 5.0:  # safety margin
            break
        
        # Use a different seed for each restart
        rng = random.Random(12345 + restart * 1000 + len(sequence))
        
        chain = _simulated_annealing(sequence, dimension, rng, time_per_restart)
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
