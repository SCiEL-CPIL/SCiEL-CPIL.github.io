"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

class _SpatialGrid:
    """Spatial hash for O(1) neighbor lookups."""
    def __init__(self, chain):
        self.grid = {p: i for i, p in enumerate(chain)}
    
    def get(self, point):
        return self.grid.get(point)
    
    def set(self, point, index):
        self.grid[point] = index
    
    def remove(self, point):
        self.grid.pop(point, None)
    
    def contains(self, point):
        return point in self.grid

def _valid_growth_start(sequence, dimension, rng):
    """Generate a random self-avoiding walk that fits the sequence length."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = _SpatialGrid(chain)
    
    for i in range(1, n):
        # Get valid moves from last position
        last = chain[-1]
        moves = [site for site in _neighbours(last) if not occupied.contains(site)]
        
        if not moves:
            # Dead end, try to backtrack? For simplicity, restart this growth.
            # In a production system, we'd backtrack. Here, we just fail this attempt.
            return None
        
        # Bias towards H-H contacts if possible
        best_move = None
        best_score = -1
        for site in moves:
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10
            # Add small random noise to avoid local minima in growth
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_move = site
        
        if best_move is None:
            best_move = rng.choice(moves)
            
        occupied.set(best_move, i)
        chain.append(best_move)
        
    return chain

def _pivot_move(sequence, chain, i, rng):
    """Perform a pivot move at index i."""
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0: # Rotate around X
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1: # Rotate around Y
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else: # Rotate around Z
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Compute energy delta
    # We need to update the grid to compute energy delta efficiently?
    # For simplicity in SA, we can just compute full energy if the move is accepted often,
    # or use a delta calculation. Given the overhead of delta calc in Python,
    # and that SA accepts many bad moves, let's try to compute delta quickly.
    
    # Old contacts involving suffix
    old_grid = _SpatialGrid(chain)
    new_grid = _SpatialGrid(new_chain)
    
    old_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                j = old_grid.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
                    
    new_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j = new_grid.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
                    
    delta = new_contacts - old_contacts
    return new_chain, delta

def _reversal_move(sequence, chain, i, j, rng):
    """Perform a reversal move on segment [i, j]."""
    n = len(chain)
    if i < 0 or j >= n or i > j:
        return None
    
    # Extract segment
    seg = chain[i:j+1]
    
    # Reverse it
    new_seg = seg[::-1]
    
    # Construct new chain
    new_chain = chain[:i] + new_seg + chain[j+1:]
    
    # Check self-avoidance
    # The reversed segment is internally valid (it was a valid SAW before, so reversed is too).
    # We only need to check intersections with the rest of the chain.
    # Rest of chain: chain[:i] and chain[j+1:]
    # Note: The endpoints of the segment (i-1 and j+1) are fixed.
    # The new segment starts at chain[i-1] (same as old chain[i] connected to it? No.)
    # Old: ... chain[i-1] - chain[i] ...
    # New: ... chain[i-1] - chain[j] ...
    # Wait, the bond between i-1 and i must remain a unit step.
    # In a reversal, the bond (i-1, i) becomes (i-1, j).
    # So we must ensure that chain[i-1] and chain[j] are neighbors.
    # Similarly, chain[j] and chain[j+1] must be neighbors.
    
    if i > 0:
        p1 = chain[i-1]
        p2 = new_seg[0] # which is chain[j]
        if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
            return None
            
    if j < n - 1:
        p1 = new_seg[-1] # which is chain[i]
        p2 = chain[j+1]
        if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
            return None
            
    # Check collisions with prefix and suffix
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])
    
    for p in new_seg:
        if p in prefix_set or p in suffix_set:
            return None
            
    # Calculate energy delta
    old_grid = _SpatialGrid(chain)
    new_grid = _SpatialGrid(new_chain)
    
    # Contacts involving the segment [i, j]
    old_contacts = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                j_idx = old_grid.get(q)
                if j_idx is not None and abs(j_idx - idx) > 1 and sequence[j_idx] == 'H':
                    old_contacts += 1
                    
    new_contacts = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j_idx = new_grid.get(q)
                if j_idx is not None and abs(j_idx - idx) > 1 and sequence[j_idx] == 'H':
                    new_contacts += 1
                    
    delta = new_contacts - old_contacts
    return new_chain, delta

def _simulated_annealing(sequence, chain, rng, deadline):
    """Simulated Annealing with Pivot and Reversal moves."""
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    t_start = 50.0
    t_end = 0.1
    # Cooling rate to reach t_end by deadline
    # Approx steps: (deadline - start_time) * 1000 (assume 1ms per step, optimistic)
    # Let's just use a fixed number of steps based on time
    max_steps = int((deadline - start_time) * 2000) # 2000 steps per second
    if max_steps < 100: max_steps = 100
    
    alpha = (t_end / t_start) ** (1.0 / max_steps)
    temperature = t_start
    
    step = 0
    while step < max_steps and time.time() < deadline:
        # Choose move type
        if rng.random() < 0.7:
            # Pivot
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        else:
            # Reversal
            # Segment length 2 to n//2
            max_len = n // 2
            if max_len < 2:
                i = rng.randint(1, n - 1)
                result = _pivot_move(sequence, current_chain, i, rng)
            else:
                i = rng.randint(1, n - max_len - 1)
                j = i + rng.randint(2, max_len)
                result = _reversal_move(sequence, current_chain, i, j, rng)
        
        if result is None:
            step += 1
            continue
            
        new_chain, delta = result
        new_energy = current_energy - delta
        
        # Acceptance criterion
        if delta > 0 or (temperature > 0 and rng.random() < math.exp(delta / temperature)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temperature *= alpha
        step += 1
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 325.0 # 5s safety margin
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    n = len(sequence)
    if n == 0: return []
    if n == 1: return [[0, 0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Generate multiple diverse initial conformations
    # We will try to generate 5-10 good starting points
    num_starts = min(5, max(1, int(remaining / 5.0)))
    
    for k in range(num_starts):
        if time.time() > deadline - 1.0: break
        start_time_k = time.time()
        chain = _valid_growth_start(sequence, dimension, rng)
        if chain is None:
            continue
            
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # If we have a very good energy, we can spend less time on other starts
        # But for now, let's just get the best start
        
    if best_chain is None:
        # Fallback: straight line (valid, but high energy)
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        
    # Phase 2: Simulated Annealing
    # Allocate remaining time to SA
    sa_deadline = _GLOBAL_DEADLINE - 1.0
    if time.time() < sa_deadline:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, sa_deadline)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Return as list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
