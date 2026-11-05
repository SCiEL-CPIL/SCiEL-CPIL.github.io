"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Deadline guard: keep a safety margin before the 210s wall-clock kill.
_DEADLINE = 210.0 - 15.0  # 195s safety margin
_START_TIME = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy: -1 per non-consecutive H-H lattice contact."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            j = where.get((nx, ny))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _grow_sequence(sequence, dimension, rng):
    """
    Randomized greedy chain growth with contact-aware scoring.
    Returns a self-avoiding walk (list of tuples) or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        x, y = last
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            site = (nx, ny)
            if site in occupied:
                continue
            # Score: prefer new H-H contacts, then free space
            gain = 0.0
            if sequence[i] == 'H':
                # Count H residues in neighbours that are not the previous residue
                for qx, qy in ((nx+1,ny),(nx-1,ny),(nx,ny+1),(nx,ny-1)):
                    j = occupied.get((qx, qy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1.0
            # Free space heuristic
            free = 0
            for qx, qy in ((nx+1,ny),(nx-1,ny),(nx,ny+1),(nx,ny-1)):
                if (qx, qy) not in occupied:
                    free += 1
            score = gain + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda t: -t[0])
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_moves=5000, rng=None):
    """
    Local search using single-residue pivot moves (relocating one residue 
    to an adjacent free site) and 3-residue flips.
    Returns improved chain and energy.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    no_improve = 0
    moves = 0
    max_no_improve = 200
    
    while moves < max_moves and no_improve < max_no_improve:
        # Try a random pivot move: move residue i to an adjacent free site
        # that is not adjacent to residue i-1 or i+1 (to maintain connectivity)
        # Actually, for a simple local search, we try to move a single residue
        # to a neighbouring free position, but we must ensure the chain remains
        # connected. A safer approach: try 3-residue flips or single-residue 
        # moves that don't break connectivity.
        
        # For simplicity and correctness, use a "kink" move: try to move 
        # residue i (not first or last) to one of its 4 neighbours, but only 
        # if that site is free and the new position is still adjacent to 
        # residue i-1 and i+1... that's too restrictive.
        
        # Better: use a "pivot" where we rotate a 3-residue segment.
        # Or simply: try moving residue i to a free neighbour, then check 
        # if the chain is still valid (self-avoiding and connected).
        
        # Given time constraints, let's use a simpler approach: 
        # Try all possible single-residue moves and accept if energy improves.
        
        i = rng.randint(1, n - 2)  # Not first or last
        x, y = current_chain[i]
        improved = False
        
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            new_pos = (nx, ny)
            # Check if new_pos is free (not occupied by any other residue)
            if new_pos in occupied:
                continue
            # Check connectivity: new_pos must be adjacent to current_chain[i-1]
            px, py = current_chain[i-1]
            if abs(px - nx) + abs(py - ny) != 1:
                continue
            # And new_pos must be adjacent to current_chain[i+1]
            qx, qy = current_chain[i+1]
            if abs(qx - nx) + abs(qy - ny) != 1:
                continue
            
            # Try the move
            old_pos = current_chain[i]
            current_chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i
            
            new_energy = _energy(sequence, current_chain)
            if new_energy < current_energy:
                current_energy = new_energy
                no_improve = 0
                improved = True
                moves += 1
                break
            else:
                # Revert
                current_chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
                moves += 1
                no_improve += 1
        
        if not improved:
            # Try a different strategy: 3-residue flip
            i = rng.randint(1, n - 3)
            p0 = current_chain[i]
            p1 = current_chain[i+1]
            p2 = current_chain[i+2]
            
            # Try flipping the middle residue
            x1, y1 = p1
            for nx, ny in ((x1+1,y1),(x1-1,y1),(x1,y1+1),(x1,y1-1)):
                new_p1 = (nx, ny)
                if new_p1 in occupied:
                    continue
                # Check adjacency with p0 and p2
                x0, y0 = p0
                x2, y2 = p2
                if abs(x0 - nx) + abs(y0 - ny) != 1:
                    continue
                if abs(x2 - nx) + abs(y2 - ny) != 1:
                    continue
                
                current_chain[i+1] = new_p1
                del occupied[p1]
                occupied[new_p1] = i+1
                
                new_energy = _energy(sequence, current_chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    no_improve = 0
                    improved = True
                    moves += 1
                    break
                else:
                    current_chain[i+1] = p1
                    del occupied[new_p1]
                    occupied[p1] = i+1
                    moves += 1
                    no_improve += 1
    
    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple randomized greedy growths followed by local search.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start with unfolded energy (0 contacts)
    
    # Number of restarts: scale with sequence length
    # For small sequences, do more restarts; for large, fewer
    if n <= 30:
        num_restarts = 300
    elif n <= 50:
        num_restarts = 200
    elif n <= 80:
        num_restarts = 100
    else:
        num_restarts = 50
    
    for restart in range(num_restarts):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow_sequence(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Local search to improve
        if _remaining_time() > 10.0:
            chain, energy = _local_search(sequence, chain, max_moves=2000, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
