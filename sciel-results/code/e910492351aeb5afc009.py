# EVOLVE-BLOCK-START
import random
import time
import sys

def _neighbours(point):
    """Generate all 6 neighbours of a point on the 3D lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the energy of a chain (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _random_grow(sequence, dimension, rng):
    """
    Grow a random self-avoiding walk, biased towards hydrophobic contacts.
    Returns None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score this move
            score = rng.random()
            
            # Prefer moves that create H-H contacts
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 5.0
            
            # Slight preference for moves that don't trap the chain
            # (count free neighbours of the candidate site)
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate with some randomness
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # With 80% probability pick the best, otherwise pick randomly from top 3
        if rng.random() < 0.8 or len(candidates) <= 1:
            best_site = candidates[0][1]
        else:
            top_k = min(3, len(candidates))
            top_candidates = [c[1] for c in candidates[:top_k]]
            best_site = rng.choice(top_candidates)
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_search(sequence, chain, rng, time_limit=5.0):
    """
    Local search using single-residue moves and segment reversals.
    Uses simulated annealing-style acceptance to escape local minima.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.95
    step = 0
    
    while time.time() - start_time < time_limit:
        step += 1
        
        # Choose move type: 70% single, 30% segment reversal
        if rng.random() < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)  # Don't move ends
            old_pos = current_chain[i]
            
            candidates = []
            for site in _neighbours(old_pos):
                if site in occupied:
                    # Check if it's the same residue
                    if occupied[site] == i:
                        continue
                    continue
                
                # Check if moving here creates self-intersection
                # (it won't since site is not occupied)
                
                # Compute energy change
                delta = 0
                
                # Remove old contacts involving residue i
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if j > i + 1 and sequence[j] == 'H':
                                delta += 1  # Removing a contact increases energy (less negative)
                            elif j < i - 1 and sequence[j] == 'H':
                                delta += 1
                
                # Add new contacts involving residue i at new position
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if j > i + 1 and sequence[j] == 'H':
                                delta -= 1  # Adding a contact decreases energy
                            elif j < i - 1 and sequence[j] == 'H':
                                delta -= 1
                
                # Check if new position conflicts with neighbours in chain
                # (already checked by occupancy)
                
                candidates.append((delta, site))
            
            if not candidates:
                continue
            
            # Pick random candidate or best
            if rng.random() < 0.8:
                # Pick best (most negative delta)
                candidates.sort(key=lambda x: x[0])
                delta, new_pos = candidates[0]
            else:
                delta, new_pos = rng.choice(candidates)
            
            # Accept or reject based on Metropolis criterion
            if delta < 0 or rng.random() < pow(sys.float_info.max, -delta / temperature):
                # Accept move
                # Update occupied
                del occupied[old_pos]
                occupied[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Segment reversal
            if n < 4:
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 1, n - 1)
            
            # Check if reversal would cause self-intersection
            segment = current_chain[i:j+1]
            reversed_segment = segment[::-1]
            
            # Check if any position in reversed_segment (except endpoints which stay)
            # conflicts with occupied positions
            valid = True
            for k in range(1, len(reversed_segment) - 1):
                pos = reversed_segment[k]
                idx = i + k
                if pos in occupied and occupied[pos] != idx:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute energy change for reversal
            delta = 0
            old_segment = current_chain[i:j+1]
            
            # Remove old contacts within the segment and with outside
            for k in range(len(old_segment)):
                idx = i + k
                if sequence[idx] == 'H':
                    for q in _neighbours(old_segment[k]):
                        jj = occupied.get(q)
                        if jj is not None and jj != idx:
                            # Check if this contact is within segment
                            if i <= jj <= j:
                                if jj > idx + 1 and sequence[jj] == 'H':
                                    delta += 1
                                elif jj < idx - 1 and sequence[jj] == 'H':
                                    delta += 1
            
            # Add new contacts
            for k in range(len(reversed_segment)):
                idx = i + k
                if sequence[idx] == 'H':
                    for q in _neighbours(reversed_segment[k]):
                        jj = occupied.get(q)
                        if jj is not None and jj != idx:
                            if i <= jj <= j:
                                if jj > idx + 1 and sequence[jj] == 'H':
                                    delta -= 1
                                elif jj < idx - 1 and sequence[jj] == 'H':
                                    delta -= 1
            
            # Apply reversal if accepted
            if delta < 0 or rng.random() < pow(sys.float_info.max, -delta / temperature):
                # Update occupied
                for k in range(len(old_segment)):
                    idx = i + k
                    del occupied[old_segment[k]]
                for k in range(len(reversed_segment)):
                    idx = i + k
                    occupied[reverse_segment[k]] = idx
                
                current_chain[i:j+1] = reversed_segment
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        # Cool down
        temperature *= cooling
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42 + n)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    total_time_limit = 120  # Leave margin from 330s limit
    
    # Phase 1: Random growth with multiple restarts
    while time.time() - start_time < total_time_limit * 0.4:
        chain = _random_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 10:
        best_chain = _local_search(sequence, best_chain, rng, time_limit=min(remaining_time * 0.8, 30))
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 3: More restarts if time permits
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 5:
        while time.time() - start_time < total_time_limit * 0.95:
            chain = _random_grow(sequence, dimension, rng)
            if chain is None:
                continue
            
            # Quick local search
            if time.time() - start_time < total_time_limit * 0.9:
                chain = _local_search(sequence, chain, rng, time_limit=1.0)
            
            energy = _compute_energy(sequence, chain)
            
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
