"""HP lattice protein folding with local search and time-based budget."""
import random
import time
from collections import defaultdict


def _neighbours(point):
    """Generate 6 neighbours on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -1 for each H-H contact (non-consecutive)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng, mode=0):
    """Greedy chain growth with different heuristics."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            if mode == 0:
                score = gain + 0.1 * free + rng.random() * 0.5
            elif mode == 1:
                score = gain * 2 + 0.05 * free + rng.random() * 0.3
            else:
                score = gain + 0.2 * free + rng.random() * 0.8
            
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _grow_random(sequence, dimension, rng):
    """Random walk growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        neighbours = list(_neighbours(chain[-1]))
        rng.shuffle(neighbours)
        for site in neighbours:
            if site not in occupied:
                occupied[site] = i
                chain.append(site)
                break
        else:
            return None
    
    return chain


def _pivot_move(chain, i, j, rng):
    """Perform a pivot move: rotate segment i..j around residue i."""
    n = len(chain)
    if i >= n or j >= n or i >= j:
        return None
    
    # Build set of occupied positions for fast lookup
    occupied = set(chain)
    
    # Remove segment positions (except i)
    segment = chain[i+1:j+1]
    for p in segment:
        occupied.discard(p)
    
    # Try all 6 possible rotations of the first segment bond
    # The bond from i to i+1 can be rotated to any of the 5 other directions
    # (can't go back to i)
    p_i = chain[i]
    p_i1 = chain[i+1]
    
    # Get the direction vector
    dir_vec = [p_i1[k] - p_i[k] for k in range(len(p_i))]
    
    # Generate all 6 unit vectors, exclude the one pointing back to i
    candidates = []
    for axis in range(len(p_i)):
        for step in (-1, 1):
            v = [0] * len(p_i)
            v[axis] = step
            if v != dir_vec:
                candidates.append(v)
    
    rng.shuffle(candidates)
    
    for new_dir in candidates:
        # Try to place the segment with the new first bond direction
        new_chain = chain[:]
        new_chain[i+1] = tuple(p_i[k] + new_dir[k] for k in range(len(p_i)))
        
        # Check if new_chain[i+1] is valid (not occupied, not i)
        if new_chain[i+1] == p_i or new_chain[i+1] in occupied:
            continue
        
        # Now propagate the rest of the segment
        valid = True
        for k in range(i+2, j+1):
            # The bond direction from k-1 to k should be preserved relative to the rotation
            old_dir = [chain[k][d] - chain[k-1][d] for d in range(len(chain[0]))]
            # We need to figure out the rotation... this is complex
            # For simplicity, let's just try random placements
            break
        
        # Actually, proper pivot moves require computing the rotation
        # Let's use a simpler approach: just try to place the whole segment
        # by maintaining relative positions but rotating the first bond
        pass
    
    # Simpler approach: just try crankshaft moves instead
    return None


def _crankshaft_move(chain, i, j, rng):
    """Perform a crankshaft move: rotate segment i..j while keeping endpoints fixed."""
    n = len(chain)
    if i >= n or j >= n or i >= j or j >= n - 1:
        return None
    
    occupied = set(chain)
    
    # Remove segment
    segment = chain[i+1:j]
    for p in segment:
        occupied.discard(p)
    
    # Try all 6 directions for the first bond
    p_i = chain[i]
    p_j = chain[j]
    
    # The segment from i+1 to j-1 must be placed such that it connects p_i to p_j
    # This is complex; let's try a simpler local move instead
    
    return None


def _local_search(sequence, chain, rng, time_limit=0.1):
    """Local search with simple moves."""
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Try a random local move: pick a random residue and try to move it
        i = rng.randint(1, n - 2)  # Don't move endpoints
        p = chain[i]
        neighbours = list(_neighbours(p))
        rng.shuffle(neighbours)
        
        for new_p in neighbours:
            if new_p in chain:
                continue
            
            # Check if moving residue i to new_p maintains self-avoidance
            # and doesn't create conflicts
            new_chain = chain[:]
            new_chain[i] = new_p
            
            # Verify self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Verify unit steps
            valid = True
            for k in range(n - 1):
                diff = sum(abs(new_chain[k+1][d] - new_chain[k][d]) for d in range(len(new_chain[0])))
                if diff != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                current_energy = new_energy
                chain = new_chain
                break
    
    return best_chain


def _simulated_annealing(sequence, chain, rng, time_limit=0.1):
    """Simulated annealing for local optimization."""
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 5.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random residue to move
        i = rng.randint(1, n - 2)
        p = chain[i]
        neighbours = list(_neighbours(p))
        rng.shuffle(neighbours)
        
        moved = False
        for new_p in neighbours:
            if new_p in chain:
                continue
            
            new_chain = chain[:]
            new_chain[i] = new_p
            
            if len(set(new_chain)) != n:
                continue
            
            valid = True
            for k in range(n - 1):
                diff = sum(abs(new_chain[k+1][d] - new_chain[k][d]) for d in range(len(new_chain[0])))
                if diff != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < (1.0 / (1.0 + (1.0 - new_energy) / (temperature + 0.1))):
                current_energy = new_energy
                chain = new_chain
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                moved = True
                break
        
        if not moved:
            temperature *= cooling
        
        if temperature < 0.01:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Fold a protein sequence on the cubic lattice."""
    n = len(sequence)
    
    # Set a deadline: use most of the 330s budget but leave safety margin
    # The evaluator kills at 330s wall-clock, so we should finish well before
    deadline = time.time() + 320  # Leave 10s safety margin
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 1  # Start with high energy (less negative is worse)
    
    # Phase 1: Generate diverse initial conformations
    max_initial = 5000
    initial_count = 0
    
    while initial_count < max_initial and time.time() < deadline - 60:
        initial_count += 1
        
        # Use different growth strategies
        if initial_count % 3 == 0:
            chain = _grow_random(sequence, dimension, rng)
        elif initial_count % 3 == 1:
            chain = _grow_greedy(sequence, dimension, rng, mode=0)
        else:
            chain = _grow_greedy(sequence, dimension, rng, mode=1)
        
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
    
    # If no valid chain found, create a straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Local search on best conformation
    if time.time() < deadline - 30:
        time_per_search = min(0.05, (deadline - 30 - time.time()) / max(1, 100))
        chain = _local_search(sequence, best_chain, rng, time_limit=time_per_search)
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Phase 3: More restarts with local search if time permits
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        
        # Generate a new initial conformation
        if restart_count % 2 == 0:
            chain = _grow_greedy(sequence, dimension, rng, mode=2)
        else:
            chain = _grow_greedy(sequence, dimension, rng, mode=0)
        
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        # If this is good enough, do local search
        if energy < best_energy + 5 and time.time() < deadline - 5:
            time_left = deadline - 5 - time.time()
            if time_left > 0.01:
                chain = _local_search(sequence, chain, rng, time_limit=min(0.02, time_left * 0.5))
                energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    return [list(p) for p in best_chain]
