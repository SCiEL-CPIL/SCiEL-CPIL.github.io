"""HP lattice protein folding using simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time tracking
_START_TIME = time.time()
_TIME_LIMIT = 190.0  # Safety margin for 210s limit

def _neighbors(point):
    """Yield 4-neighbors of a point."""
    x, y = point
    yield (x+1, y)
    yield (x-1, y)
    yield (x, y+1)
    yield (x, y-1)

def _compute_energy(sequence, chain):
    """Compute the energy of the conformation. Energy = -contacts."""
    n = len(chain)
    # Map coordinates to indices
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # Only need to check H residues
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for nb in _neighbors(p):
                j = pos_to_idx.get(nb)
                if j is not None:
                    # Check if not adjacent in sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            contacts += 1
    # Each contact is counted twice (once from i, once from j), so divide by 2
    return -contacts // 2

def _initial_fold(sequence, dimension):
    """Generate an initial self-avoiding walk using a simple greedy/random strategy."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Directions: Right, Down, Left, Up
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    current_dir = 0
    
    for i in range(1, n):
        found = False
        # Try to continue in current direction, then turn left, then right
        # This creates a compact spiral-like structure initially
        for offset in [0, 1, 3]: 
            d_idx = (current_dir + offset) % 4
            dx, dy = dirs[d_idx]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                current_dir = d_idx
                found = True
                break
        
        if not found:
            # If stuck, try all directions
            for d_idx in range(4):
                dx, dy = dirs[d_idx]
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    current_dir = d_idx
                    found = True
                    break
            
            if not found:
                # Dead end, return partial or restart? 
                # For simplicity, if we get stuck, we return what we have if it's full length, 
                # but this greedy might fail for longer sequences. 
                # Let's use a more robust random walk for initialization if this fails.
                pass

    if len(chain) < n:
        # Fallback: Simple random walk
        chain = [(0, 0)]
        occupied = {(0, 0)}
        for i in range(1, n):
            attempts = 0
            while attempts < 100:
                d = random.randint(0, 3)
                dx, dy = dirs[d]
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    break
                attempts += 1
            if attempts >= 100:
                # Failed to grow, return straight line as last resort (always valid)
                return [(i, 0) for i in range(n)]
        
    return chain

def _apply_pivot(chain, k, rotation):
    """
    Apply a pivot move at index k with given rotation (0, 90, 180, 270).
    Returns the new chain if valid, None if collision.
    """
    n = len(chain)
    if k >= n:
        return None
    
    # The part to rotate is chain[k:]
    # Pivot point is chain[k-1] if k>0, else we rotate around origin? 
    # Standard pivot: rotate chain[k:] around chain[k-1] (if k>0) or around (0,0) if k==0?
    # Usually pivot is defined by rotating the tail. Let's rotate around chain[k].
    # Actually, standard pivot moves rotate the subchain starting at k around the bond (k-1, k) or just point k.
    # Let's rotate the segment chain[k:] around the point chain[k].
    # Wait, if we rotate around chain[k], chain[k] stays fixed.
    
    pivot = chain[k]
    new_tail = []
    for i in range(k, n):
        x, y = chain[i]
        dx = x - pivot[0]
        dy = y - pivot[1]
        
        if rotation == 90:
            # (x, y) -> (-y, x)
            new_x = -dy
            new_y = dx
        elif rotation == 180:
            # (x, y) -> (-x, -y)
            new_x = -dx
            new_y = -dy
        elif rotation == 270:
            # (x, y) -> (y, -x)
            new_x = dy
            new_y = -dx
        else: # 0
            new_x = dx
            new_y = dy
            
        new_tail.append((pivot[0] + new_x, pivot[1] + new_y))
        
    new_chain = chain[:k] + new_tail
    
    # Check self-avoidance
    # The head part chain[:k] is unchanged.
    # We only need to check if new_tail intersects with chain[:k-1] (since chain[k] is shared/pivot)
    # Actually, chain[k] is in both. We need to ensure new_tail[1:] does not intersect chain[:k-1].
    # And new_tail must be self-avoiding (it is, since it's a rigid rotation of a SAW).
    
    head_set = set(chain[:k-1]) if k > 1 else set()
    # If k==0, head is empty.
    # If k==1, head is chain[0].
    
    for p in new_tail[1:]: # Skip the pivot point itself
        if p in head_set:
            return None
            
    # Also check if new_tail intersects with itself? No, rigid rotation preserves internal distances.
    # But we must ensure the bond between k-1 and k is valid? 
    # The bond (k-1, k) is fixed. The bond (k, k+1) changes direction.
    # We just need to ensure no overlap.
    
    return new_chain

def _apply_creep(chain, i):
    """
    Try to move residue i to a random empty neighbor.
    Returns new chain if valid, None otherwise.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None # Don't move ends usually, or handle carefully. Let's allow internal only for simplicity.
    
    x, y = chain[i]
    neighbors = list(_neighbors((x, y)))
    random.shuffle(neighbors)
    
    # Build set of occupied positions excluding i
    occupied = set(chain)
    occupied.remove((x, y))
    
    for nb in neighbors:
        if nb not in occupied:
            new_chain = chain[:]
            new_chain[i] = nb
            return new_chain
            
    return None

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Check time
    if time.time() - _START_TIME > _TIME_LIMIT:
        return [[i, 0] for i in range(n)]

    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts depends on time available
    # Let's do a fixed number of iterations per restart, or time-based.
    # Given 10 sequences, we have ~19s per sequence.
    # Let's allocate 18s per sequence.
    
    seq_start = time.time()
    seq_time_limit = 18.0
    
    # Initial fold
    current_chain = _initial_fold(sequence, dimension)
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.99 # Cooling rate
    
    # Precompute H indices for faster energy diff? 
    # For now, full recompute is O(N). N<=100. 100 ops is fast.
    # We can do 10,000 steps per second easily.
    
    steps = 0
    while time.time() - seq_start < seq_time_limit:
        # Choose move
        if random.random() < 0.7:
            # Pivot
            k = random.randint(1, n - 1)
            rot = random.choice([90, 180, 270])
            new_chain = _apply_pivot(current_chain, k, rot)
        else:
            # Creep
            i = random.randint(1, n - 2)
            new_chain = _apply_creep(current_chain, i)
            
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        # Acceptance criterion
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher T or break?
            # Let's restart the annealing schedule
            T = 1.0
            # Optional: restart chain? No, keep exploring.
            
        steps += 1
        
        # Safety check for global time
        if time.time() - _START_TIME > _TIME_LIMIT - 1.0:
            break
            
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
