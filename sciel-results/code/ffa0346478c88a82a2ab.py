"""
HP lattice protein folding using a constructive random-growth local search.

Strategy:
1. Instead of pre-computing a fixed skeleton (like Boustrophedon) and then refining it,
   we grow the chain one residue at a time.
2. At each step, we try all 4 possible directions for the next residue.
3. We use a scoring function that balances:
   - Immediate energy gain (new H-H contacts formed).
   - "Openness" (number of valid next moves available for the new position).
   - Randomness to escape local minima and explore different conformations.
4. We perform multiple restarts with different random seeds to find the best conformation.
5. We use a fast incremental energy calculation to speed up the search.
"""

import random
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Generate the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Compute HP contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow_chain(sequence, rng, max_steps=100000):
    """
    Grow a chain using a biased random walk.
    Returns a list of lattice points representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at (0, 0)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Directions: Right, Down, Left, Up
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # For the first step, pick a random direction
    first_dir = rng.choice(directions)
    chain.append((first_dir[0], first_dir[1]))
    occupied[(first_dir[0], first_dir[1])] = 1
    
    current_energy = 0
    
    for i in range(2, n):
        best_score = -float('inf')
        best_pos = None
        best_new_energy = 0
        
        last_pos = chain[-1]
        prev_pos = chain[-2]
        
        # Determine valid directions (avoid backtracking and collisions)
        valid_dirs = []
        for dx, dy in directions:
            # Avoid backtracking
            if (dx, dy) == (-prev_pos[0] + last_pos[0], -prev_pos[1] + last_pos[1]):
                continue
            
            new_pos = (last_pos[0] + dx, last_pos[1] + dy)
            
            # Check collision
            if new_pos in occupied:
                continue
            
            # Check if this move creates a "dead end" (no further moves possible)
            # This is a heuristic to avoid trapping the chain
            # Count valid next moves from new_pos
            next_moves = 0
            for ndx, ndy in directions:
                # Avoid backtracking to last_pos
                if (ndx, ndy) == (-dx, -dy):
                    continue
                next_pos = (new_pos[0] + ndx, new_pos[1] + ndy)
                if next_pos not in occupied:
                    next_moves += 1
            
            if next_moves == 0:
                continue # Dead end, skip this move
            
            valid_dirs.append((new_pos, next_moves))
        
        if not valid_dirs:
            # No valid moves, break (should not happen if we avoid dead ends)
            break
        
        # Score each valid move
        for new_pos, next_moves in valid_dirs:
            # Calculate immediate energy gain
            new_energy_gain = 0
            if sequence[i] == 'H':
                # Check neighbors of new_pos for existing H residues
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            new_energy_gain += 1
            
            # Scoring function:
            # 1. Prioritize energy gains
            # 2. Prioritize moves that leave more options (next_moves)
            # 3. Add randomness
            score = new_energy_gain * 10.0 + next_moves * 0.5 + rng.random() * 2.0
            
            if score > best_score:
                best_score = score
                best_pos = new_pos
                best_new_energy = new_energy_gain
        
        if best_pos is None:
            break
        
        # Move to best_pos
        chain.append(best_pos)
        occupied[best_pos] = i
        current_energy -= best_new_energy
        
        # Optional: If we have too few moves left, we might get stuck.
        # But the dead-end avoidance helps.
        
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using a constructive random-growth local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Time budget management
    start_time = time.time()
    time_limit = 10.0  # Conservative per-sequence limit
    
    best_chain = None
    best_energy = 0  # Initialize to 0 (worst case is no contacts)
    
    # Number of restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(12345 + restart * 7919 + n)
        
        # Grow a new chain
        chain = _grow_chain(sequence, rng)
        
        if len(chain) < n:
            # If the chain is incomplete, try to extend it or discard it
            # For simplicity, discard incomplete chains
            continue
        
        # Calculate energy
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no valid chain was found (should be rare), return a simple linear chain
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
