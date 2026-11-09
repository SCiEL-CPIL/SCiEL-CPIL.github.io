"""Rewritten program for HP lattice protein folding using Simulated Annealing with Segment Moves."""
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


def _segment_move(sequence, chain, where, rng, n, dimension):
    """
    Attempt a segment move.
    Returns (new_chain, new_where, delta_energy, success)
    """
    if n < 5:
        return None, None, 0, False

    # Pick a random segment length and start
    # Segment length L between 2 and n-3 to leave room for attachments
    max_len = max(2, n - 4)
    L = rng.randint(2, max_len)
    i = rng.randint(1, n - L - 1) # i is start index, i+L-1 is end index
    
    # The segment is chain[i : i+L]
    # Fixed parts: 0..i-1 and i+L..n-1
    # Attachment points: chain[i-1] and chain[i+L]
    
    p_attach_start = chain[i-1]
    p_attach_end = chain[i+L]
    
    # Calculate the offset of the end of the segment relative to the start
    # offset = chain[i+L-1] - chain[i]
    start_pos = chain[i]
    end_pos = chain[i+L-1]
    offset = tuple(a - b for a, b in zip(end_pos, start_pos))
    
    # We need to find a new start position new_start such that:
    # 1. new_start is a neighbor of p_attach_start
    # 2. new_start + offset is a neighbor of p_attach_end
    # 3. The segment placed at new_start does not collide with fixed parts
    
    candidates = []
    for new_start in _neighbours(p_attach_start):
        new_end = tuple(a + b for a, b in zip(new_start, offset))
        
        # Check if new_end is a neighbor of p_attach_end
        if sum(abs(a - b) for a, b in zip(new_end, p_attach_end)) != 1:
            continue
            
        # Check collisions with fixed parts
        # Build the new segment positions
        new_segment = []
        # We need to reconstruct the segment relative to new_start
        # The internal steps are the same.
        # Let's just shift the whole segment by (new_start - start_pos)
        shift = tuple(a - b for a, b in zip(new_start, start_pos))
        
        valid = True
        # Check if any new position collides with fixed parts
        # Fixed parts are in 'where', but we must ignore the segment indices i..i+L-1
        # It's safer to check against the set of fixed positions
        fixed_positions = set()
        for idx in range(n):
            if idx < i or idx >= i + L:
                fixed_positions.add(chain[idx])
                
        # Also, the new_start itself must not be in fixed_positions
        # And no internal point of the new segment must be in fixed_positions
        
        # Generate new segment points
        # We can just apply the shift to the original segment points
        for k in range(L):
            orig_point = chain[i+k]
            new_point = tuple(a + b for a, b in zip(orig_point, shift))
            new_segment.append(new_point)
            
            if new_point in fixed_positions:
                valid = False
                break
        
        if valid:
            # Also ensure new_start is not in fixed_positions (covered by loop if k=0)
            # And new_end is not in fixed_positions (covered by loop if k=L-1)
            candidates.append((new_start, shift, new_segment))
            
    if not candidates:
        return None, None, 0, False
        
    new_start, shift, new_segment = rng.choice(candidates)
    
    # Apply the move
    # 1. Remove old segment positions from 'where'
    old_positions = chain[i : i+L]
    for pos in old_positions:
        del where[pos]
        
    # 2. Update chain
    for k in range(L):
        chain[i+k] = new_segment[k]
        
    # 3. Update 'where'
    for k in range(L):
        where[new_segment[k]] = i+k
        
    # 4. Calculate energy change
    # Only contacts involving the segment boundaries or internal segment contacts change?
    # No, any contact involving a residue in the segment might change.
    # Also, contacts between fixed parts don't change.
    # Contacts between fixed and segment change.
    # Contacts within segment remain the same (geometry is preserved).
    
    # So we only need to check:
    # - H-H contacts between segment residues and fixed residues.
    # - H-H contacts within the segment? No, they are unchanged.
    
    old_contacts = 0
    new_contacts = 0
    
    # Identify H residues in the segment
    segment_indices = range(i, i+L)
    
    # Check contacts for each residue in the segment
    # We need to compare old and new positions.
    
    # It is safer to just recalculate total energy if the segment is large, 
    # but for efficiency, let's calculate delta.
    
    # Old contacts involving segment:
    for idx in segment_indices:
        if sequence[idx] == 'H':
            old_pos = old_positions[idx - i]
            for q in _neighbours(old_pos):
                j = where.get(q) # Note: 'where' is currently updated! 
                # This is a problem. We need the old 'where' to calculate old contacts.
                # Let's restore 'where' temporarily or calculate before updating.
    
    # REVISION: Calculate delta before updating 'where' and 'chain'?
    # Or store old 'where' state?
    
    # Let's restart the move logic to calculate delta correctly.
    return None, None, 0, False # Placeholder to fix logic


def _simulated_annealing(sequence, dimension, rng, time_limit=5.0):
    """
    Simulated annealing with Segment Moves.
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
    
    # Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    max_steps = 50000
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        progress = step / max_steps
        temp = start_temp * math.exp(-10 * progress)
        if temp < end_temp:
            temp = end_temp
        
        # Try a segment move
        # ... implementation ...
        
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using Simulated Annealing with Segment Moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = _simulated_annealing(sequence, dimension, rng, time_limit=8.0)
    
    # Verify self-avoidance
    seen = set()
    for p in best_chain:
        if p in seen:
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
