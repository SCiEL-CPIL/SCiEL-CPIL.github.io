import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 310.0  # Safety margin before 330s kill

def _neighbours(point):
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
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            if q in where:
                j = where[q]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _energy_incremental(sequence, chain, old_chain, pivot_idx):
    """
    Compute energy change after a pivot move at pivot_idx.
    Only residues at pivot_idx+1..end moved. We need to recompute contacts
    involving any H residue that moved or any H residue adjacent to moved positions.
    For simplicity and correctness, we recompute full energy but with optimized lookup.
    """
    return _energy(sequence, chain)

def _beam_search(sequence, dimension, beam_width, time_limit):
    """
    Beam search to build initial conformations step by step.
    Returns list of (chain, energy) for top beam_width partial/full chains.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [([(0, 0, 0)], 0)]
    
    # Precompute H positions for faster energy calc
    h_indices = set(i for i, c in enumerate(sequence) if c == 'H')
    
    start = (0, 0, 0)
    initial_chain = [start]
    initial_occ = {start}
    initial_energy = 0
    beam = [(initial_chain, initial_occ, initial_energy)]
    
    for step in range(1, n):
        new_beam = []
        for chain, occ, energy in beam:
            last = chain[-1]
            for q in _neighbours(last):
                if q in occ:
                    continue
                new_chain = chain + [q]
                new_occ = occ | {q}
                
                # Incremental energy: only check if new residue creates H-H contact
                new_energy = energy
                if sequence[step] == 'H':
                    for nb in _neighbours(q):
                        if nb in occ:
                            # Find which index this neighbor is
                            # We need to check if it's H and not adjacent in chain
                            # Since we're building sequentially, the neighbor at position nb
                            # is at some earlier index j. If j is H and |step - j| > 1, it's a contact.
                            # We need a way to find j. Let's use a pos-to-idx map for current chain.
                            # But that's expensive. Alternative: just do full energy for beam search.
                            pass
                
                # For beam search, do full energy computation (beam width is small)
                # But optimize: only compute if last residue is H
                if sequence[step] == 'H':
                    # Check if new position creates any H-H contact with existing H residues
                    # Build a quick lookup for this chain
                    pos_to_idx = {}
                    for idx, pos in enumerate(chain):
                        pos_to_idx[pos] = idx
                    for nb in _neighbours(q):
                        if nb in pos_to_idx:
                            j = pos_to_idx[nb]
                            if abs(step - j) > 1 and sequence[j] == 'H':
                                new_energy = energy - 1  # One more contact
                # Also check if existing H residues now have new contacts? No, only new residue moves.
                
                new_beam.append((new_chain, new_occ, new_energy))
        
        if not new_beam:
            break
        
        # Sort by energy (lower is better)
        new_beam.sort(key=lambda x: x[2])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Check time
        if time.time() > time_limit:
            break
    
    return [(chain, energy) for chain, occ, energy in beam]

def _pivot_move(chain, i, axis, angle):
    """Rotate chain[i+1:] around chain[i] parallel to axis."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    pivot = chain[i]
    new_chain = list(chain[:i+1])
    
    for p in chain[i+1:]:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        rv = _rotate_vector(v, axis, angle)
        new_chain.append((rv[0] + pivot[0], rv[1] + pivot[1], rv[2] + pivot[2]))
    
    return new_chain

def _rotate_vector(v, axis, angle):
    """Rotate vector v around axis by angle (1=90, 2=180, 3=270 degrees)."""
    x, y, z = v
    if axis == 0:  # x-axis
        if angle == 1:
            return (x, -z, y)
        elif angle == 2:
            return (x, -y, -z)
        else:
            return (x, z, -y)
    elif axis == 1:  # y-axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, -z)
        else:
            return (-z, y, x)
    else:  # z-axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        else:
            return (-y, x, z)

def _is_valid_after_pivot(chain, i):
    """Check if chain is self-avoiding after pivot move at index i. Only check tail vs head."""
    n = len(chain)
    if n <= 2:
        return True
    
    head_set = set(chain[:i+1])
    for p in chain[i+1:]:
        if p in head_set:
            return False
    return True

def _simulated_annealing(sequence, chain, time_limit, start_temp=5.0, cool_rate=0.999):
    """Simulated annealing with pivot moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temp = start_temp
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random pivot index
        i = random.randint(1, n - 2)
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])
        
        new_chain = _pivot_move(current_chain, i, axis, angle)
        if new_chain is None:
            continue
        
        # Check validity: only need to check if tail intersects head
        if not _is_valid_after_pivot(new_chain, i):
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temp *= cool_rate
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Global deadline guard
    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining < 2.0:
        # Not enough time, return a simple valid chain
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time: use up to 80% of remaining for this sequence
    time_per_sequence = min(remaining * 0.8, 10.0)
    if time_per_sequence < 1.0:
        time_per_sequence = max(remaining * 0.9, 0.5)
    
    # Determine beam width based on chain length
    if n <= 20:
        beam_width = 50
    elif n <= 50:
        beam_width = 30
    elif n <= 80:
        beam_width = 20
    else:
        beam_width = 15
    
    # Phase 1: Beam search to get good initial conformations
    beam_time = time_per_sequence * 0.4
    beam_results = _beam_search(sequence, dimension, beam_width, beam_time)
    
    if not beam_results:
        # Fallback: simple linear chain
        return [[i, 0, 0] for i in range(n)]
    
    # Phase 2: Local search on top candidates
    # Take top 3 candidates
    candidates = [chain for chain, energy in beam_results[:3]]
    
    best_chain = None
    best_energy = 0
    
    sa_time_per_candidate = (time_per_sequence * 0.5) / len(candidates)
    
    for candidate in candidates:
        if time.time() > _GLOBAL_DEADLINE - 2.0:
            break
        
        result = _simulated_annealing(sequence, candidate, sa_time_per_candidate)
        e = _energy(sequence, result)
        
        if best_chain is None or e < best_energy:
            best_energy = e
            best_chain = result
    
    # Phase 3: Additional refinement on the best result with remaining time
    elapsed = time.time() - (_GLOBAL_DEADLINE - time_per_sequence)
    remaining_time = time_per_sequence - elapsed
    if remaining_time > 1.0 and best_chain is not None:
        result = _simulated_annealing(sequence, best_chain, remaining_time, start_temp=3.0, cool_rate=0.9995)
        e = _energy(sequence, result)
        if e < best_energy:
            best_energy = e
            best_chain = result
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
