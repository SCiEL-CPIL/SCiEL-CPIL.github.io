"""HP lattice protein folding on the cubic lattice - Monte Carlo with Pivot Moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _rotate_90(point, axis, direction):
    """
    Rotate a point (relative to origin) by 90 degrees around the given axis.
    axis: 0 (x), 1 (y), or 2 (z)
    direction: 1 or -1
    """
    x, y, z = point
    if axis == 0: # x-axis
        if direction == 1:
            return (x, z, -y)
        else:
            return (x, -z, y)
    elif axis == 1: # y-axis
        if direction == 1:
            return (-z, y, x)
        else:
            return (z, y, -x)
    else: # z-axis
        if direction == 1:
            return (y, -x, z)
        else:
            return (-y, x, z)

def _initial_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    for _ in range(n - 1):
        current = chain[-1]
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if abs(dx) + abs(dy) + abs(dz) != 1:
                        continue
                    new_pos = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if new_pos not in occupied:
                        candidates.append(new_pos)
        if not candidates:
            return None
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain

def _compute_energy(sequence, chain):
    """Compute energy of the chain."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in [(p[0]+1, p[1], p[2]), (p[0]-1, p[1], p[2]), 
                  (p[0], p[1]+1, p[2]), (p[0], p[1]-1, p[2]), 
                  (p[0], p[1], p[2]+1), (p[0], p[1], p[2]-1)]:
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _pivot_move(sequence, chain, rng):
    """
    Attempt a pivot move. Returns (new_chain, new_energy, valid) 
    or (None, None, False) if invalid.
    """
    n = len(chain)
    if n < 2:
        return None, None, False
    
    # Choose pivot index
    k = rng.randint(0, n - 2)
    pivot = chain[k]
    
    # Extract tail
    tail = chain[k+1:]
    
    # Choose rotation
    axis = rng.randint(0, 2)
    direction = rng.choice([-1, 1])
    
    # Translate tail to origin
    tail_rel = [(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]) for p in tail]
    
    # Rotate
    tail_rot = [_rotate_90(p, axis, direction) for p in tail_rel]
    
    # Translate back
    new_tail = [(p[0] + pivot[0], p[1] + pivot[1], p[2] + pivot[2]) for p in tail_rot]
    
    # Check validity: 
    # 1. First point of new tail must be adjacent to pivot (it is, by rotation properties)
    # 2. New tail must not intersect the fixed part chain[0:k+1]
    # 3. New tail must be self-avoiding
    
    fixed_set = set(chain[:k+1])
    new_chain = chain[:k+1] + new_tail
    new_occupied = set(new_chain)
    
    if len(new_occupied) != n:
        return None, None, False
    
    new_energy = _compute_energy(sequence, new_chain)
    return new_chain, new_energy, True

def _local_move(sequence, chain, rng):
    """
    Attempt a local move: move the last residue.
    """
    n = len(chain)
    if n < 2:
        return None, None, False
    
    # Move the last residue
    i = n - 1
    current = chain[i]
    prev = chain[i-1]
    
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if abs(dx) + abs(dy) + abs(dz) != 1:
                    continue
                new_pos = (prev[0] + dx, prev[1] + dy, prev[2] + dz)
                if new_pos == current:
                    continue
                # Check if new_pos is occupied by any other part of the chain
                if new_pos in set(chain[:-1]):
                    continue
                candidates.append(new_pos)
    
    if not candidates:
        return None, None, False
    
    new_pos = rng.choice(candidates)
    new_chain = chain[:-1] + [new_pos]
    new_energy = _compute_energy(sequence, new_chain)
    return new_chain, new_energy, True

def _monte_carlo(sequence, dimension, time_budget, rng):
    n = len(sequence)
    
    # Initial state
    chain = None
    attempts = 0
    while chain is None and attempts < 100:
        chain = _initial_saw(n, rng)
        attempts += 1
    if chain is None:
        # Fallback straight line
        chain = [(i, 0, 0) for i in range(n)]
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 50.0
    t_final = 0.1
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Temperature schedule
        progress = min(1.0, elapsed / time_budget)
        temp = t0 + (t_final - t0) * progress
        
        # Choose move type: 80% Pivot, 20% Local
        if rng.random() < 0.8:
            new_chain, new_energy, valid = _pivot_move(sequence, chain, rng)
        else:
            new_chain, new_energy, valid = _local_move(sequence, chain, rng)
        
        if not valid:
            continue
        
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using Monte Carlo with Pivot Moves.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(54321 + len(sequence))
    
    # Time allocation
    base_time = 4.0
    time_budget = base_time + (n / 100.0) * 4.0
    time_budget = min(time_budget, 15.0)
    
    chain, energy = _monte_carlo(sequence, dimension, time_budget, rng)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
