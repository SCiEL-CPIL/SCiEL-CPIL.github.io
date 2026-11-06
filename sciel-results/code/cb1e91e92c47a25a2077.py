"""HP lattice protein folding using Simulated Annealing with Reversal Moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 4 neighbors of a point on the 2D square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _calculate_energy(sequence, chain):
    """Calculate the total energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk to start the annealing."""
    n = len(sequence)
    if n == 1:
        return [(0, 0)]
    
    # Try a few random walks, if fail, use a simple snake
    for _ in range(10):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            moved = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    moved = True
                    break
            if not moved:
                break
        if len(chain) == n:
            return chain
            
    # Fallback: Simple snake
    chain = []
    y = 0
    x = 0
    for i in range(n):
        chain.append((x, y))
        # Zig-zag pattern to ensure self-avoidance
        if i % 2 == 0:
            x += 1
        else:
            y += 1
            x += 1
    return chain

def _try_reversal(sequence, chain, i, j, pos_to_idx, is_h):
    """
    Attempt to reverse the segment chain[i:j+1].
    Returns (new_chain, delta_energy, success)
    """
    n = len(chain)
    if i >= j:
        return None, 0, False
    
    # Extract segment
    segment = chain[i:j+1]
    
    # New positions for the segment (reversed)
    new_segment = segment[::-1]
    
    # Check collision with outside
    # We need to check if any point in new_segment is occupied by an index outside [i, j]
    for k, new_pos in enumerate(new_segment):
        idx = pos_to_idx.get(new_pos)
        if idx is not None:
            # If the position is occupied by an index in [i, j], it's fine (it's part of the moving segment)
            # BUT, we must ensure we don't collide with the *old* positions of the segment that are being vacated?
            # No, the positions are being vacated. The new positions are what matter.
            # If idx is in [i, j], it means the new position is currently occupied by a residue that is *also* moving.
            # This is valid as long as the mapping is bijective.
            # We only care if a new position collides with a *static* (outside) residue.
            if not (i <= idx <= j):
                return None, 0, False
    
    # Collision check passed. Now calculate Delta Energy.
    
    # Old Energy contribution from segment to outside
    e_old = 0
    for k in range(i, j+1):
        if is_h[k]:
            p = chain[k]
            for q in _neighbours(p):
                idx_q = pos_to_idx.get(q)
                if idx_q is not None and not (i <= idx_q <= j):
                    if is_h[idx_q]:
                        e_old += 1
                        
    # New Energy contribution from segment to outside
    e_new = 0
    for k in range(i, j+1):
        if is_h[k]:
            p = new_segment[k-i] # New position of residue k
            for q in _neighbours(p):
                idx_q = pos_to_idx.get(q)
                if idx_q is not None and not (i <= idx_q <= j):
                    if is_h[idx_q]:
                        e_new += 1
                        
    delta_e = e_new - e_old
    
    # Construct new chain
    new_chain = chain[:i] + new_segment + chain[j+1:]
    
    return new_chain, delta_e, True

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    is_h = [c == 'H' for c in sequence]
    
    # Deadline: 180s limit, use 170s to be safe
    start_time = time.time()
    deadline = start_time + 170
    
    best_chain = None
    best_energy = 0
    
    # Initial best: straight line or simple fold
    rng = random.Random(42)
    
    init_chain = _initial_chain(sequence, dimension, rng)
    best_chain = init_chain
    best_energy = _calculate_energy(sequence, best_chain)
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.99
    
    # Number of restarts
    while time.time() < deadline:
        # Restart with a new random initial chain
        current_chain = _initial_chain(sequence, dimension, rng)
        current_energy = _calculate_energy(sequence, current_chain)
        
        # If this initial is already better, update
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = current_chain
            
        # SA Loop
        T = T_start
        # Limit steps per restart
        steps_per_restart = 20000 
        if n > 50:
            steps_per_restart = 10000
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Pick i, j for reversal
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            
            # Ensure segment length is at least 2
            if j - i < 1:
                continue
                
            # Build pos_to_idx for current chain
            pos_to_idx = {p: idx for idx, p in enumerate(current_chain)}
            
            new_chain, delta_e, success = _try_reversal(
                sequence, current_chain, i, j, pos_to_idx, is_h
            )
            
            if not success:
                continue
                
            # Acceptance
            if delta_e < 0:
                current_chain = new_chain
                current_energy += delta_e
            else:
                # Metropolis
                if T > 1e-6:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        current_chain = new_chain
                        current_energy += delta_e
                        
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
            # Cool down
            T *= cooling_rate
            
    # Convert tuples to lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
