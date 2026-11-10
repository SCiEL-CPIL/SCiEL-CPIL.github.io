"""
HP Lattice Protein Folding using Simulated Annealing with Pivot Moves.
"""
import random
import time
import math

# Global deadline guard
_START_TIME = time.time()
_MAX_TIME = 100.0  # Safety margin before 120s/180s limit

def _time_left():
    return _MAX_TIME - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield the 4 neighbors of a point on a 2D square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """
    Calculate the energy of a conformation.
    Energy = - (number of non-bonded H-H contacts).
    """
    # Map position to index
    pos_to_idx = {pos: i for i, pos in enumerate(chain)}
    contacts = 0
    for i, pos in enumerate(chain):
        if sequence[i] == 'H':
            for nb in _neighbours(pos):
                j = pos_to_idx.get(nb)
                if j is not None:
                    # Ensure they are not adjacent in the chain (bonded)
                    if abs(i - j) > 1:
                        # Count each pair once (i < j)
                        if i < j and sequence[j] == 'H':
                            contacts += 1
    return -contacts

def _grow(sequence, dimension, rng, bias=0.5):
    """
    Generate an initial self-avoiding walk using a randomized greedy approach.
    bias controls how much we prefer sites with more free space vs H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _neighbours(last):
            if nb not in occupied:
                # Calculate a score for this candidate
                score = rng.random()
                
                # Prefer sites that create H-H contacts
                if sequence[i] == 'H':
                    for nb2 in _neighbours(nb):
                        if nb2 in occupied:
                            # Check if the occupant is H and not adjacent in chain
                            # We don't have index easily here without a map, 
                            # but for initial growth, simple proximity is a good heuristic.
                            # Actually, let's just count neighbors in occupied to estimate compactness
                            pass
                
                # Prefer sites with more free neighbors (to avoid trapping)
                free_count = sum(1 for nb2 in _neighbours(nb) if nb2 not in occupied)
                score += free_count * 0.1
                
                # If current residue is H, prefer being close to existing Hs?
                # Simple heuristic: if sequence[i] is H, add a small bonus for being surrounded
                if sequence[i] == 'H':
                    h_count = 0
                    for nb2 in _neighbours(nb):
                        if nb2 in occupied:
                            # We don't know if it's H without a map, so we skip specific H-check
                            # to keep it fast. The local search will fix this.
                            pass
                            
                candidates.append((score, nb))
        
        if not candidates:
            return None # Failed to grow
        
        # Sort by score descending and pick top one (or random from top few for diversity)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pick from the top 3 to maintain some diversity
        k = min(3, len(candidates))
        chosen = rng.choice(candidates[:k])
        chain.append(chosen[1])
        occupied.add(chosen[1])
        
    return chain

def _local_search(sequence, chain, rng, time_limit=5.0):
    """
    Perform local search using single-step moves and pivot moves.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # Cooling schedule
    T = 10.0
    cooling_rate = 0.95
    min_T = 0.01
    
    start_time = time.time()
    
    while T > min_T and (time.time() - start_time) < time_limit and _time_left() > 1.0:
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Single step move (pivot of length 1 is just moving the end? No, standard move is moving a residue)
            # Let's implement a "pivot" move: reverse a segment [i, j]
            # A standard pivot move in protein folding is often a rotation or reversal.
            # Reversal is easy to implement and check.
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Try reversing chain[i:j+1]
            new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
            
            # Check self-avoidance
            # Optimization: Only check the moved segment against the rest?
            # For simplicity and safety, check full set. 
            # To speed up, we can use a set.
            if len(set(new_chain)) == n:
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain
        else:
            # End move: move the last residue to a different adjacent spot
            last_idx = n - 1
            last_pos = current_chain[last_idx]
            prev_pos = current_chain[last_idx - 1]
            
            # Find a valid new position for the last residue
            valid_moves = []
            for nb in _neighbours(prev_pos):
                if nb == last_pos:
                    continue
                if nb not in set(current_chain[:-1]): # Check against all except the last one
                    valid_moves.append(nb)
            
            if valid_moves:
                new_pos = rng.choice(valid_moves)
                new_chain = current_chain[:-1] + [new_pos]
                
                # Self-avoidance check for the whole chain is implicitly done by the check above
                # but we must ensure new_pos doesn't collide with any other.
                # The check `nb not in set(current_chain[:-1])` ensures it doesn't collide with 0..n-2.
                # It is distinct from last_pos by construction.
                
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain
        
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if not sequence:
        return []
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Strategy:
    # 1. Generate a few diverse initial chains.
    # 2. Apply local search to each.
    # 3. Keep the best.
    
    num_initials = 5
    local_search_time = 10.0  # seconds per local search pass
    
    for i in range(num_initials):
        if _time_left() < local_search_time + 1.0:
            break
            
        # Generate initial chain
        chain = _grow(sequence, dimension, rng, bias=0.5)
        
        if chain is None:
            # Fallback: try a simple spiral or line if growth fails
            # A line is always valid
            chain = [(j, 0) for j in range(n)]
            
        # Run local search
        # Allocate more time for later iterations if we have time?
        # Or just fixed time.
        optimized_chain = _local_search(sequence, chain, rng, time_limit=local_search_time)
        
        energy = _energy(sequence, optimized_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = optimized_chain
            
        # If we found a very good energy (e.g., close to theoretical min), we could stop early?
        # Theoretical min is hard to know without lookup.
        
    if best_chain is None:
        # Absolute fallback
        best_chain = [(j, 0) for j in range(n)]
        
    return [list(p) for p in best_chain]
