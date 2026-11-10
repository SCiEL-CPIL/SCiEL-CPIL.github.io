"""HP lattice protein folding with simulated annealing and pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for q in _neighbours(chain[i]):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a segment of the chain around a pivot point.
    Returns (new_chain, success) where success is True if the move is valid."""
    n = len(chain)
    if n < 4:
        return None, False
    
    # Choose pivot point (not first or last)
    pivot = rng.randint(1, n - 2)
    
    # Choose segment length to rotate
    max_len = min(pivot, n - pivot - 1)
    if max_len < 1:
        return None, False
    seg_len = rng.randint(1, max_len)
    
    # The segment to rotate is chain[pivot-seg_len+1 : pivot+1]
    # Pivot is at chain[pivot]
    # We rotate the segment ending at the pivot
    
    # Get the segment
    start_idx = pivot - seg_len + 1
    segment = chain[start_idx:pivot+1]
    
    # The fixed part before the segment
    before = chain[:start_idx]
    # The fixed part after the pivot
    after = chain[pivot+1:]
    
    # Try all 6 possible rotations of the segment
    # The segment's first point is connected to before[-1]
    # We need to rotate the segment so that it starts from a different direction
    
    # Get the direction from before[-1] to segment[0]
    if before:
        prev = before[-1]
        d0 = tuple(segment[0][a] - prev[a] for a in range(3))
    else:
        return None, False
    
    # Generate all 6 possible new directions
    directions = []
    for axis in range(3):
        for sign in (-1, 1):
            d = [0, 0, 0]
            d[axis] = sign
            d = tuple(d)
            if d != d0:
                directions.append(d)
    
    if not directions:
        return None, False
    
    # Pick a random new direction
    new_d0 = rng.choice(directions)
    
    # Rotate the entire segment
    # The segment is a path; we need to apply a rotation that maps d0 to new_d0
    # Find a rotation matrix that maps d0 to new_d0
    
    # Simple approach: find the axis of rotation
    # d0 and new_d0 are unit vectors along axes
    # We can use a series of 90-degree rotations
    
    # Actually, let's use a different approach: just try translating the segment
    # to start in the new direction and check validity
    
    # The segment starts at segment[0] and goes in direction d0
    # We want it to start at (prev + new_d0) and follow the same relative path
    
    # Compute the relative offsets within the segment
    base = segment[0]
    offsets = [tuple(segment[i][a] - base[a] for a in range(3)) for i in range(seg_len)]
    
    # New base position
    new_base = tuple(prev[a] + new_d0[a] for a in range(3))
    
    # New segment positions
    new_segment = [tuple(new_base[a] + offsets[i][a] for a in range(3)) for i in range(seg_len)]
    
    # Build the new chain
    new_chain = before + new_segment + after
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, False
        seen.add(p)
    
    # Check unit steps
    for i in range(len(new_chain) - 1):
        diff = sum(abs(new_chain[i+1][a] - new_chain[i][a]) for a in range(3))
        if diff != 1:
            return None, False
    
    return new_chain, True


def _initial_fold(sequence, dimension):
    """Create an initial self-avoiding walk. Use a simple spiral/helix pattern."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Try to grow in a helical pattern
    directions = [
        (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    current_dir = 0
    for i in range(1, n):
        placed = False
        for offset in range(6):
            d_idx = (current_dir + offset) % 6
            d = directions[d_idx]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                current_dir = d_idx
                placed = True
                break
        if not placed:
            # Backtrack or just stop
            break
    
    # If we didn't place all residues, fill in with a simple pattern
    while len(chain) < n:
        # Try all 6 directions
        placed = False
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                placed = True
                break
        if not placed:
            # Give up and use a straight line from here
            last = chain[-1]
            for j in range(len(chain), n):
                chain.append((last[0] + (j - len(chain)), last[1], last[2]))
            break
    
    return chain


def _simulate_annealing(sequence, dimension, chain, time_budget, rng):
    """Run simulated annealing on the given initial chain."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    iterations = 0
    max_iterations = 500000  # Safety limit
    
    while T > T_end and iterations < max_iterations:
        # Try a pivot move
        new_chain, success = _pivot_move(chain, rng)
        if not success:
            iterations += 1
            T *= cooling_rate
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept or reject
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        iterations += 1
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence using simulated annealing with time budget."""
    start_time = time.time()
    deadline = start_time + 300  # Safety margin before 330s kill
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    # Count H residues to estimate target
    h_count = sequence.count('H')
    
    # Multiple restarts with different initial configurations
    best_chain = None
    best_energy = 0  # Start with unfolded energy
    
    num_restarts = 10
    time_per_restart = (300 - 30) / num_restarts  # 30s safety margin
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > 300 - 30:  # Safety margin
            break
        
        # Create initial fold
        if restart == 0:
            # First restart: use the structured initial fold
            chain = _initial_fold(sequence, dimension)
        else:
            # Subsequent restarts: random self-avoiding walks
            chain = None
            for _ in range(100):
                chain = _random_saw(sequence, dimension, rng)
                if chain is not None and len(chain) == n:
                    break
            if chain is None or len(chain) != n:
                chain = _initial_fold(sequence, dimension)
        
        # Run simulated annealing
        time_budget = time_per_restart - (time.time() - start_time)
        if time_budget < 5:
            break
        
        sa_chain, sa_energy = _simulate_annealing(sequence, dimension, chain, time_budget, rng)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
        
        # Check time
        if time.time() - start_time > 300 - 30:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]


def _random_saw(sequence, dimension, rng):
    """Generate a random self-avoiding walk of the given length."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for i in range(1, n):
        available = []
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                available.append(new_pos)
        
        if not available:
            return None
        
        next_pos = rng.choice(available)
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain
# EVOLVE-BLOCK-END
