"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None

def _neighbours(point):
    """Yield 6 neighbors of a point in 3D lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50, max_depth=None, time_limit=1.0):
    """
    Beam search to generate initial conformations.
    Returns a list of chains (top beam_width) and their energies.
    """
    if max_depth is None:
        max_depth = len(sequence)
        
    n = len(sequence)
    if n == 0:
        return []
        
    start = (0, 0, 0)
    initial_chain = [start]
    initial_occ = {start}
    initial_energy = 0
    beam = [(initial_chain, initial_occ, initial_energy)]
    
    start_time = time.time()
    
    for step in range(1, max_depth):
        new_beam = []
        for chain, occ, energy in beam:
            last = chain[-1]
            for q in _neighbours(last):
                if q in occ:
                    continue
                new_chain = chain + [q]
                new_occ = occ | {q}
                # Incremental energy update would be faster, but full is okay for small N in beam
                # Actually, for N up to 100, O(N) energy calc inside beam search (width 50) is 50*100 = 5000 ops per step.
                # 100 steps * 5000 = 500k ops. This is fast in Python? Maybe.
                # Let's try to be smarter. Only H residues contribute.
                # But for simplicity and robustness, let's just compute full energy if n < 50, else maybe skip beam search?
                # No, beam search is the key improvement. Let's optimize _energy slightly or just trust it.
                # 26 sequences. If we spend 1s on beam search, that's 26s total.
                
                new_energy = _energy(sequence, new_chain)
                new_beam.append((new_chain, new_occ, new_energy))
                
        if not new_beam:
            break
            
        # Sort by energy (lower is better)
        new_beam.sort(key=lambda x: x[2])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Time check
        if time.time() - start_time > time_limit:
            break
            
    return [b[0] for b in beam]

def _rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector v around a given axis (0,1,2) by 90, 180, or 270 degrees.
    Returns the new vector.
    """
    x, y, z = v
    if axis == 0: # X axis
        if angle == 1: # 90 deg
            return (x, -z, y)
        elif angle == 2: # 180 deg
            return (x, -y, -z)
        elif angle == 3: # 270 deg
            return (x, z, -y)
    elif axis == 1: # Y axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, z)
        elif angle == 3:
            return (-z, y, x)
    elif axis == 2: # Z axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        elif angle == 3:
            return (-y, x, z)
    return v

def _pivot_move(sequence, chain, i, rng):
    """
    Pivot move: Rotate suffix chain[i:] around chain[i-1].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    # Choose rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_suffix = []
    for p in suffix:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
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
    
    # Calculate energy delta
    # Contacts involving indices >= i.
    # Internal contacts of suffix are preserved.
    # We need to check contacts between suffix and prefix.
    
    prefix_map = {}
    for idx in range(i):
        prefix_map[chain[idx]] = idx
    
    old_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _segment_move(sequence, chain, i, j, rng):
    """
    Segment move: Rotate chain[i+1:j+1] around chain[i].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    pivot = chain[i]
    segment = chain[i+1:j+1]
    
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_segment = []
    for p in segment:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_segment.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i+1]) # Includes pivot
    suffix_set = set(chain[j+1:])
    seen = set()
    for p in new_segment:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i+1] + new_segment + chain[j+1:]
    
    # Energy delta
    static_map = {}
    for idx in range(0, i+1):
        static_map[chain[idx]] = idx
    for idx in range(j+1, n):
        static_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _is_valid_chain(chain):
    """
    Check if the chain is self-avoiding and has unit steps.
    """
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if dist != 1:
                return False
    return True

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot and Segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    initial_temp = 5.0
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Mix of Pivot and Segment moves
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        else:
            # Segment
            # Pick a segment length
            seg_len = rng.randint(2, min(10, n - 1))
            i = rng.randint(0, n - 1 - seg_len)
            j = i + seg_len
            result = _segment_move(sequence, current_chain, i, j, rng)
        
        if result is None:
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
            
        # Cooling
        temperature *= 0.999
        
        if temperature < 0.01:
            if no_improve_count > max_no_improve:
                temperature = initial_temp * 0.5
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.1
                no_improve_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 320.0  # Safety margin
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    # Allocate time
    # Beam search is fast. Let's give it 1 second.
    # Rest for SA.
    beam_time = 1.0
    sa_deadline = start_time + remaining - 0.5
    
    # 1. Beam Search Initialization
    beam_chains = _beam_search(sequence, dimension, beam_width=20, time_limit=beam_time)
    
    if not beam_chains:
        # Fallback to random growth or straight line
        chain = [(i, 0, 0) for i in range(n)]
        best_chain, best_energy = _simulated_annealing(sequence, chain, rng, sa_deadline)
    else:
        # Pick the best chain from beam search
        # They are already sorted by energy in _beam_search? No, I returned just chains.
        # Let's re-evaluate or just take the first one (which was best in the last step).
        # Actually, _beam_search returns [b[0] for b in beam] where beam is sorted.
        # So beam_chains[0] is the best.
        
        # We can run SA on the top 1-2 candidates if time permits, but to keep it simple and robust:
        # Run SA on the best candidate.
        best_chain, best_energy = _simulated_annealing(sequence, beam_chains[0], rng, sa_deadline)
        
        # If time remains, maybe try the 2nd best?
        # For now, stick to one to ensure we finish.
        
    # Validation
    if not _is_valid_chain(best_chain):
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Convert to lists of ints
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
