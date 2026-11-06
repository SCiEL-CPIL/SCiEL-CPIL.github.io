import random
import time

# EVOLVE-BLOCK-START

_T0 = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _neighbours(point):
    # Unroll for 2D for speed
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _energy(seq, chain):
    """Compute energy: -number of H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(seq):
    """Randomized greedy growth to get a valid initial chain."""
    n = len(seq)
    if n == 0:
        return []
    
    # Start at (0,0)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    last_dir = None
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        # Potential moves: 4 directions
        # Prefer continuing in same direction or turning, avoid immediate backtracking unless necessary
        moves = [(x+1, y, 'E'), (x-1, y, 'W'), (x, y+1, 'N'), (x, y-1, 'S')]
        # Filter out occupied
        valid_moves = []
        for mx, my, d in moves:
            if (mx, my) not in occupied:
                # Avoid immediate backtrack (U-turn) unless no other option
                if last_dir and d == last_dir and random.random() > 0.1:
                    continue
                valid_moves.append((mx, my, d))
        
        if not valid_moves:
            # Backtrack search or restart? For simplicity, if stuck, we might return None 
            # but greedy growth with randomization rarely gets stuck for N=100 if we allow U-turns
            # Let's allow U-turns if nothing else works
            valid_moves = [(mx, my, d) for mx, my, d in moves if (mx, my) not in occupied]
            if not valid_moves:
                return None
            # Pick one
            mx, my, d = random.choice(valid_moves)
        else:
            # Weighted choice: prefer non-U-turns
            # Simple random choice from valid is fine for initialization
            mx, my, d = random.choice(valid_moves)
            
        chain.append((mx, my))
        occupied[(mx, my)] = i
        last_dir = d
        
    return chain

def _local_search(seq, chain, time_limit):
    """
    Perform local search using reversal moves.
    Returns the best chain found.
    """
    if not chain:
        return chain
    
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(seq, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute H positions for faster checks? 
    # Actually, _energy is O(N). N=100. 1000 moves = 100k ops. Very fast.
    
    rng = random.Random()
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit and _remaining() > 5:
        # Choose a random segment to reverse
        # Length of segment at least 2
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Perform reversal: chain[i:j+1] becomes reversed
        # New chain
        # Check validity
        # The reversed part is current_chain[i:j+1][::-1]
        # We need to ensure these new positions don't collide with current_chain[0:i] or current_chain[j+1:]
        
        # Optimization: 
        # The internal contacts of the reversed segment are preserved (just reversed order, so adjacencies remain adjacencies).
        # We only need to check:
        # 1. The new position of the first element of the reversed segment (old j) against its new neighbors (i-1 and i+1... wait)
        #    Actually, the neighbors of the reversed segment in the full chain are:
        #    Left neighbor: index i-1 (if i>0)
        #    Right neighbor: index j+1 (if j<n-1)
        #    The elements inside the reversed segment are adjacent to each other, so no internal collisions.
        #    The only potential collisions are:
        #    - New pos of old[j] (now at i) with old[i-1]
        #    - New pos of old[i] (now at j) with old[j+1]
        #    - New pos of any element in the reversed segment with the *rest* of the chain?
        #      NO. The rest of the chain is static. The reversed segment moves to new positions.
        #      We must check if ANY new position of the reversed segment is occupied by ANY part of the chain 
        #      that is NOT part of the reversed segment.
        
        # Let's construct the new positions
        rev_seg = current_chain[i:j+1][::-1]
        
        # Create a set of occupied positions by the "background" (non-reversed parts)
        # Background indices: 0..i-1 and j+1..n-1
        bg_positions = set()
        for k in range(0, i):
            bg_positions.add(current_chain[k])
        for k in range(j+1, n):
            bg_positions.add(current_chain[k])
        
        # Check if any position in rev_seg is in bg_positions
        valid = True
        for p in rev_seg:
            if p in bg_positions:
                valid = False
                break
        
        if not valid:
            continue
            
        # Also check connectivity with boundaries
        # If i > 0, current_chain[i] (new) must be adjacent to current_chain[i-1]
        # New current_chain[i] is rev_seg[0] which is old current_chain[j]
        # Is old current_chain[j] adjacent to current_chain[i-1]?
        # Not necessarily. The move is a "topological" reversal. 
        # Wait, in HP lattice folding, a "reversal" move usually implies that the segment is flipped in place.
        # Does this preserve lattice connectivity?
        # If we reverse the sequence of points, the step from point A to B becomes B to A. 
        # Since the lattice is undirected, if A and B were adjacent, B and A are adjacent.
        # So internal connectivity is preserved.
        # Boundary connectivity:
        # Left boundary: chain[i-1] and chain[i] (new).
        # Old chain[i] was adjacent to chain[i-1].
        # New chain[i] is old chain[j].
        # Is old chain[j] adjacent to chain[i-1]? NO, generally not.
        # So a simple array reversal does NOT preserve the "unit step" constraint at the boundaries!
        
        # CORRECTION:
        # A valid local search move must result in a self-avoiding walk with unit steps.
        # Reversal of a sub-chain `i..j` changes the chain to:
        # C[0..i-1] + C[j..i] + C[j+1..N-1]
        # The step between C[i-1] and C[j] must be unit.
        # The step between C[i] and C[j+1] must be unit.
        # These are NOT guaranteed by the original chain properties.
        # Therefore, simple reversal is NOT a valid move in the space of SAWs unless we verify boundary connectivity.
        # 
        # Most HP folding algorithms use "moves" that preserve connectivity, like:
        # 1. Pivot moves (rotate a sub-chain around a pivot point).
        # 2. Reversal moves are valid ONLY if the endpoints of the reversed segment were already adjacent to the new endpoints? No.
        # 
        # Actually, standard "reversal" in protein folding literature often refers to changing the *sequence* of turns, or it is a move that is checked for validity.
        # If we just reverse the list of coordinates, we break the chain unless the specific geometry allows it.
        # 
        # Let's use **Pivot Moves** or **Random Walk Regeneration** of a segment?
        # Pivot move: Choose a pivot index i. Rotate the tail C[i+1..N-1] around C[i] by 90, 180, or 270 degrees.
        # This preserves the unit step at C[i] to C[i+1].
        # We must check for self-intersection of the rotated tail with the head C[0..i].
        #
        # Pivot moves are standard and valid.
        
        # Let's implement Pivot Move instead of Reversal.
        
        # Choose pivot i from 1 to n-2 (to have a head and a tail)
        i = rng.randint(1, n - 2)
        
        pivot = current_chain[i]
        tail = current_chain[i+1:]
        
        # Try 3 rotations (90, 180, 270). 0 is current.
        angles = [1, 2, 3]
        rng.shuffle(angles)
        
        accepted = False
        for k in angles:
            # Rotate tail around pivot
            # Vector from pivot to next point in tail: v = tail[0] - pivot
            # Rotate v by 90*k degrees
            # new_tail[0] = pivot + rotated_v
            # new_tail[m] = new_tail[m-1] + (old_tail[m] - old_tail[m-1])
            
            # Calculate new tail
            new_tail = [pivot] # Placeholder, we will append
            # Actually, let's just compute the points
            
            # Initial vector
            dx = tail[0][0] - pivot[0]
            dy = tail[0][1] - pivot[1]
            
            # Rotate (dx, dy) by 90 * k
            # 90: (dy, -dx)
            # 180: (-dx, -dy)
            # 270: (-dy, dx)
            if k == 1:
                ndx, ndy = dy, -dx
            elif k == 2:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = -dy, dx
                
            new_tail = []
            cur_x, cur_y = pivot[0] + ndx, pivot[1] + ndy
            new_tail.append((cur_x, cur_y))
            
            # Propagate the rest of the tail using the original deltas
            valid = True
            occupied_head = set(current_chain[:i+1]) # Head includes pivot
            
            for m in range(1, len(tail)):
                # Delta from tail[m-1] to tail[m]
                ddx = tail[m][0] - tail[m-1][0]
                ddy = tail[m][1] - tail[m-1][1]
                cur_x += ddx
                cur_y += ddy
                new_tail.append((cur_x, cur_y))
                
                # Check collision with head
                if (cur_x, cur_y) in occupied_head:
                    valid = False
                    break
            
            # Also check if the new tail intersects itself?
            # The relative shape is preserved, so if the original tail was self-avoiding, the new tail is self-avoiding.
            # We only need to check against the head.
            
            if valid:
                # Construct new chain
                new_chain = current_chain[:i+1] + new_tail
                new_energy = _energy(seq, new_chain)
                
                if new_energy <= current_energy:
                    # Accept
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    accepted = True
                    break # Move applied, break angle loop
        
        # If no move accepted, continue loop
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using randomized growth and local search with time guard.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than -1
    
    rng = random.Random(12345 + n)
    
    # If sequence is very short, just return a valid path
    if n <= 5:
        # Generate a simple L-shape or straight line
        # Straight line is valid
        return [[i, 0] for i in range(n)]
        
    # Main search loop
    while _remaining() > 10:
        # 1. Generate initial chain
        chain = _grow(sequence)
        if chain is None:
            # Try again
            continue
            
        # 2. Local search
        # Allocate some time for this restart
        # E.g., 2 seconds or remaining time / 10
        sub_time = min(2.0, _remaining() - 5)
        
        improved_chain = _local_search(sequence, chain, sub_time)
        
        e = _energy(sequence, improved_chain)
        
        if best_chain is None or e < best_energy:
            best_chain = improved_chain
            best_energy = e
            
        # Check if we found the theoretical minimum? 
        # Max contacts for n residues is roughly (n-2) or so. 
        # If best_energy is very low, we might stop early? 
        # For now, rely on time.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END
