"""Rewritten program for HP lattice protein folding using simulated annealing with local moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (or 2^(dim-1) in dim dimensions)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _count_contacts(sequence, chain, where):
    """Count H-H non-consecutive contacts."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy_from_contacts(contacts):
    return -contacts


def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk using a simple random walk with backtracking."""
    n = len(sequence)
    if n == 1:
        return [(0,) * dimension]
    
    # Try multiple times to generate a self-avoiding walk
    for attempt in range(50):
        chain = [(0,) * dimension]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    
    # Fallback: straight line
    return [(i, 0, 0) if dimension == 3 else (i,) + (0,) * (dimension - 1) for i in range(n)]


def _simulated_annealing(sequence, dimension, rng, time_limit=5.0):
    """
    Simulated annealing with local moves (single-residue moves and small flips).
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return _initial_chain(sequence, dimension, rng)
    
    # Initialize with a random self-avoiding walk
    chain = _initial_chain(sequence, dimension, rng)
    where = {p: i for i, p in enumerate(chain)}
    
    # Compute initial energy
    contacts = _count_contacts(sequence, chain, where)
    current_energy = _energy_from_contacts(contacts)
    
    best_chain = list(chain)
    best_energy = current_energy
    best_contacts = contacts
    
    # Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    # We'll run for a fixed number of steps or until time limit
    max_steps = 50000
    start_time = time.time()
    
    for step in range(max_steps):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule (exponential cooling)
        progress = step / max_steps
        temp = start_temp * math.exp(-10 * progress)
        if temp < end_temp:
            temp = end_temp
        
        # Choose a move: single-residue move or 3-residue flip
        move_type = rng.random()
        
        if move_type < 0.7:
            # Single-residue move: pick a random interior residue and try to move it to a neighbor
            if n <= 3:
                continue
            i = rng.randint(1, n - 2)  # interior residue
            old_pos = chain[i]
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            
            moved = False
            for new_pos in neighbors:
                # Check if new position is valid
                # 1. Not occupied by another residue
                # 2. Moving residue i to new_pos doesn't create conflicts with i-1 and i+1
                
                # Check if new_pos is occupied by any other residue
                if new_pos in where and where[new_pos] != i:
                    continue
                
                # Check adjacency to neighbors i-1 and i+1
                # After move, chain[i] = new_pos, so new_pos must be adjacent to old positions of i-1 and i+1
                # Actually, we need to check that the new position is adjacent to both i-1 and i+1
                pos_im1 = chain[i - 1]
                pos_ip1 = chain[i + 1]
                
                # Check if new_pos is adjacent to pos_im1
                diff1 = [abs(a - b) for a, b in zip(new_pos, pos_im1)]
                if sum(diff1) != 1:
                    continue
                
                # Check if new_pos is adjacent to pos_ip1
                diff2 = [abs(a - b) for a, b in zip(new_pos, pos_ip1)]
                if sum(diff2) != 1:
                    continue
                
                # Also need to check that moving i doesn't cause i-1 and i+1 to be adjacent to each other
                # in a way that creates a non-consecutive contact issue? No, that's fine.
                # But we need to ensure that after the move, the chain is still self-avoiding.
                # The only potential conflict is if new_pos was occupied by i-1 or i+1, but we checked that.
                # Also, we need to make sure that i-1 and i+1 are not the same position (they aren't in a SAW).
                
                # Perform the move
                # Remove old position from where
                del where[old_pos]
                # Add new position
                where[new_pos] = i
                chain[i] = new_pos
                
                # Compute energy change
                # Old contacts involving i:
                old_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = where.get(q)
                        if j is not None and j != i:
                            # j is not i, so check if j > i+1 or j < i-1
                            if (j > i + 1 and sequence[j] == 'H') or (j < i - 1 and sequence[j] == 'H'):
                                old_contacts += 1
                
                new_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(new_pos):
                        j = where.get(q)
                        if j is not None and j != i:
                            if (j > i + 1 and sequence[j] == 'H') or (j < i - 1 and sequence[j] == 'H'):
                                new_contacts += 1
                
                # Also, moving i might affect contacts between i-1 and i+1? No, because i-1 and i+1 positions don't change.
                # But wait, we need to check if i-1 and i+1 were in contact before and are no longer, or vice versa.
                # Actually, the contact between i-1 and i+1 depends on their positions, which don't change.
                # So the only energy change is from contacts involving residue i.
                
                delta_contacts = new_contacts - old_contacts
                delta_energy = _energy_from_contacts(delta_contacts)
                
                # Accept or reject
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    # Accept
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_contacts = _count_contacts(sequence, chain, where)
                    moved = True
                    break
                else:
                    # Reject: revert
                    where[old_pos] = i
                    del where[new_pos]
                    chain[i] = old_pos
            
            if not moved:
                continue
        
        elif move_type < 0.9:
            # 3-residue flip: pick i, i+1, i+2 and try to flip them
            if n <= 4:
                continue
            i = rng.randint(1, n - 4)
            # Positions i, i+1, i+2
            p0 = chain[i]
            p1 = chain[i + 1]
            p2 = chain[i + 2]
            
            # The three residues form a path. We can try to reflect p1 about the midpoint of p0 and p2,
            # or try other configurations. A simpler approach: try to move p1 to a different neighbor of p0 that
            # is also a neighbor of p2, or try to move the whole segment.
            # Let's try: find all positions q such that q is adjacent to p0 and q is adjacent to p2, and q != p1.
            # Then move p1 to q.
            
            candidates = []
            for q in _neighbours(p0):
                if q == p1:
                    continue
                # Check if q is adjacent to p2
                diff = [abs(a - b) for a, b in zip(q, p2)]
                if sum(diff) == 1:
                    # q is adjacent to both p0 and p2
                    # Check if q is occupied by another residue
                    if q in where and where[q] not in (i, i + 1, i + 2):
                        continue
                    candidates.append(q)
            
            if candidates:
                new_p1 = rng.choice(candidates)
                
                # Perform the move: chain[i+1] = new_p1
                old_pos = p1
                new_pos = new_p1
                
                # Update where
                del where[old_pos]
                where[new_pos] = i + 1
                chain[i + 1] = new_pos
                
                # Compute energy change
                # Contacts involving i+1 change
                old_contacts = 0
                if sequence[i + 1] == 'H':
                    for q in _neighbours(old_pos):
                        j = where.get(q)
                        if j is not None and j != i + 1:
                            if (j > i + 2 and sequence[j] == 'H') or (j < i and sequence[j] == 'H'):
                                old_contacts += 1
                
                new_contacts = 0
                if sequence[i + 1] == 'H':
                    for q in _neighbours(new_pos):
                        j = where.get(q)
                        if j is not None and j != i + 1:
                            if (j > i + 2 and sequence[j] == 'H') or (j < i and sequence[j] == 'H'):
                                new_contacts += 1
                
                delta_contacts = new_contacts - old_contacts
                delta_energy = _energy_from_contacts(delta_contacts)
                
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_contacts = _count_contacts(sequence, chain, where)
                else:
                    # Revert
                    where[old_pos] = i + 1
                    del where[new_pos]
                    chain[i + 1] = old_pos
        
        else:
            # Try a larger move: pick a random segment and try to reinsert it elsewhere? 
            # For simplicity, let's just do a random restart with probability
            if rng.random() < 0.01:
                # Restart with a new random chain
                chain = _initial_chain(sequence, dimension, rng)
                where = {p: i for i, p in enumerate(chain)}
                contacts = _count_contacts(sequence, chain, where)
                current_energy = _energy_from_contacts(contacts)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_contacts = contacts
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using simulated annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    # Use simulated annealing with a time budget
    # The total time for all 26 sequences is 240s, so ~9s per sequence on average.
    # Let's use 8s per sequence to be safe.
    best_chain = _simulated_annealing(sequence, dimension, rng, time_limit=8.0)
    
    # Verify self-avoidance
    seen = set()
    for p in best_chain:
        if p in seen:
            # Fallback to straight line
            best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
            break
        seen.add(p)
    
    # Verify unit steps
    for i in range(1, n):
        diff = sum(abs(a - b) for a, b in zip(best_chain[i], best_chain[i - 1]))
        if diff != 1:
            best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
