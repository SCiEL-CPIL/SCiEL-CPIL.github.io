"""HP lattice protein folding using multi-start simulated annealing with fast local moves."""
import random
import time
import math

# --- Global time management ---
_total_start_time = None
_total_time_limit = 200.0  # Safety margin before 210s kill

def _init_time():
    global _total_start_time
    if _total_start_time is None:
        _total_start_time = time.time()

def _get_time_left():
    _init_time()
    return _total_time_limit - (time.time() - _total_start_time)

def _neighbours(point):
    """Generate 4-neighbor lattice points."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    n = len(sequence)
    if n < 3:
        return 0
    
    # Build position map
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    for i in h_positions:
        pi = chain[i]
        for q in _neighbours(pi):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    
    return -contacts

def _generate_initial_saw(n, rng, sequence):
    """Generate a self-avoiding walk with some bias toward H-H contacts."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours(last)
        rng.shuffle(neighbors)
        
        # Score each candidate
        best_site = None
        best_score = -float('inf')
        
        for site in neighbors:
            if site in occupied:
                continue
            
            score = 0.0
            # Bonus for H-H contacts if current residue is H
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0
            
            # Slight preference for more free neighbors (avoid trapping)
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            
            # Random tiebreaker
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            # Dead end: try to backtrack or just pick any free neighbor
            # For simplicity, if we get stuck, try a different strategy
            # Try all neighbors again without scoring
            for site in neighbors:
                if site not in occupied:
                    best_site = site
                    break
            
            if best_site is None:
                # Complete dead end - this shouldn't happen often
                # Generate a simple line as fallback
                return [(i, 0) for i in range(n)]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _sa_search(sequence, chain, time_limit, rng):
    """Simulated annealing with fast move generation."""
    n = len(sequence)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.5
    T_min = 0.001
    cooling_rate = 0.9999
    
    # Pre-compute which residues are H for fast checks
    is_h = [seq == 'H' for seq in sequence]
    
    step = 0
    while time.time() - start_time < time_limit:
        step += 1
        
        # Pick a random interior residue to move
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try each neighbor
        neighbors = _neighbours(old_pos)
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos == old_pos:
                continue
            if new_pos in occupied:
                continue
            
            # Check if move maintains chain connectivity
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            if _manhattan(new_pos, prev_pos) != 1:
                continue
            if _manhattan(new_pos, next_pos) != 1:
                continue
            
            # Apply move temporarily
            del occupied[old_pos]
            occupied[new_pos] = i
            current_chain[i] = new_pos
            
            # Compute new energy
            new_energy = _compute_energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Reject: revert
                current_chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
        
        if not moved:
            # Try a different index or just continue
            pass
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_chain

def _multi_start_sa(sequence, time_limit, rng):
    """Run multiple independent SA searches and return the best result."""
    n = len(sequence)
    if n == 0:
        return []
    if n < 3:
        return [list(p) for p in [(0, 0)] * n]
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts based on available time
    # Each SA run should take roughly time_limit / num_restarts
    # For short sequences, we can afford more restarts
    if n <= 20:
        num_restarts = 20
    elif n <= 40:
        num_restarts = 15
    elif n <= 60:
        num_restarts = 10
    else:
        num_restarts = 8
    
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        remaining = time_limit - (r * time_per_restart)
        if remaining <= 1.0:
            break
        
        # Generate initial chain
        chain = _generate_initial_saw(n, rng, sequence)
        
        # Run SA with slightly less time than allocated (buffer)
        sa_time = max(0.1, time_per_restart * 0.9)
        result = _sa_search(sequence, chain, sa_time, rng)
        energy = _compute_energy(sequence, result)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = result
    
    if best_chain is None:
        # Fallback: simple line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of (x, y) integer lattice points.
    """
    _init_time()
    n = len(sequence)
    if n == 0:
        return []
    
    time_left = _get_time_left()
    if time_left <= 0:
        # Return trivial solution
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    # Use most of the remaining time for this sequence
    # If this is one of 10 calls, we should use ~time_left/remaining_calls
    # But we don't know remaining calls, so use a heuristic
    # Cap at 18s per call to be safe (10 * 18 = 180s, leaving 30s buffer)
    time_for_this = min(time_left, 18.0)
    
    if time_for_this <= 0.5:
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    rng = random.Random(42 + len(sequence))
    
    # Run multi-start SA
    best_chain = _multi_start_sa(sequence, time_for_this, rng)
    
    return best_chain
