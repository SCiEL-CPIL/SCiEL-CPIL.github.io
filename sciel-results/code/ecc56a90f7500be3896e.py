"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

_TOTAL_START = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin: 295s instead of 330s
_PER_SEQ_BUDGET = _TOTAL_BUDGET / 26.0 * 0.92  # ~10.5s per sequence

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _neighbours_set(point):
    """Return the 6 neighbors of a point as a set."""
    return frozenset(tuple(list(point) + [s] if i != axis else tuple(list(point[:i]) + [point[i] + s] + list(point[i+1:])) 
                          for axis in range(3) for s in (-1, 1) for i in range(3) if True)
                   if False else {tuple([point[0]+(s if a==0 else 0), point[1]+(s if a==1 else 0), point[2]+(s if a==2 else 0)]) for a in range(3) for s in (-1,1)})

def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    Prefers sites with new H-H contacts, then free space, then random noise.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain: new H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Calculate free space: number of unoccupied neighbors
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: gain is primary, free space is secondary, noise for diversity
            score = gain * 10.0 * bias + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score, site = max(options)
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """
    Calculate the energy of the chain.
    Energy = - (number of H-H contacts on non-adjacent lattice neighbors).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, deadline):
    """
    Hill-climbing local search using single-residue moves.
    A move is valid if residue i is moved to a site s such that:
    1. s is a neighbor of chain[i-1] and chain[i+1]
    2. s is not occupied by any other residue
    3. s != chain[i]
    """
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    # Precompute is_h for faster access
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of residues to avoid bias
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() >= deadline:
                break
                
            old_pos = best[i]
            prev_pos = best[i - 1]
            next_pos = best[i + 1]
            
            # Find candidate sites: neighbors of prev_pos AND next_pos
            # that are not occupied by anyone else
            candidates = []
            for site in _neighbours(prev_pos):
                if site == old_pos:
                    continue
                # Check if site is neighbor of next_pos
                # Manhattan distance must be 1
                if sum(abs(site[d] - next_pos[d]) for d in range(3)) == 1:
                    # Check if site is free
                    if site not in occupied:
                        candidates.append(site)
            
            if not candidates:
                continue
            
            # Try moving to each candidate
            for site in candidates:
                # Calculate energy delta
                delta_contacts = 0
                
                # If residue i is H, it contributes to contacts
                if is_h[i]:
                    # Lost contacts: neighbors of old_pos that are H and occupied by j != i
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts -= 1
                    
                    # Gained contacts: neighbors of site that are H and occupied by j != i
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts += 1
                
                # If delta_contacts > 0, energy decreases (improves)
                if delta_contacts > 0:
                    # Apply move
                    del occupied[old_pos]
                    occupied[site] = i
                    best[i] = site
                    best_e -= delta_contacts
                    improved = True
                    break
    
    return best, best_e

def _simulated_annealing(sequence, chain, rng, deadline, t_max=5.0, t_min=0.1, alpha=0.995):
    """
    Simulated annealing local search using single-residue moves.
    """
    n = len(chain)
    best = list(chain)
    current = list(chain)
    best_e = _energy(sequence, best)
    current_e = best_e
    occupied = {p: i for i, p in enumerate(current)}
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    t = t_max
    while t > t_min and time.time() < deadline:
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        prev_pos = current[i - 1]
        next_pos = current[i + 1]
        
        candidates = []
        for site in _neighbours(prev_pos):
            if site == old_pos:
                continue
            if sum(abs(site[d] - next_pos[d]) for d in range(3)) == 1:
                if site not in occupied:
                    candidates.append(site)
        
        if not candidates:
            t *= alpha
            continue
        
        site = rng.choice(candidates)
        
        # Calculate energy delta
        delta_contacts = 0
        if is_h[i]:
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i:
                    if is_h[j]:
                        delta_contacts -= 1
            for q in _neighbours(site):
                j = occupied.get(q)
                if j is not None and j != i:
                    if is_h[j]:
                        delta_contacts += 1
        
        delta_e = -delta_contacts
        
        # Accept with probability exp(-delta_e / t) if delta_e > 0
        if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / t):
            # Apply move
            del occupied[old_pos]
            occupied[site] = i
            current[i] = site
            current_e += delta_e
            
            if current_e < best_e:
                best = list(current)
                best_e = current_e
        
        t *= alpha
    
    return best, best_e

def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Many greedy growths with varying bias
    2. Local search and simulated annealing on the best candidates
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Time budget for this sequence
    current_time = time.time()
    elapsed_global = current_time - _TOTAL_START
    remaining_global = _TOTAL_BUDGET - elapsed_global
    
    seq_deadline = min(current_time + _PER_SEQ_BUDGET, _TOTAL_START + _TOTAL_BUDGET)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growths with varying bias
    if n <= 48:
        n_restarts = 400
    elif n <= 80:
        n_restarts = 250
    else:
        n_restarts = 120
    
    # First, do standard greedy
    for restart in range(n_restarts // 2):
        if time.time() >= seq_deadline - 3.0:
            break
        
        chain = _grow(sequence, dimension, rng, bias=1.0)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Then, do more aggressive greedy (higher bias)
    for restart in range(n_restarts // 2):
        if time.time() >= seq_deadline - 2.0:
            break
        
        chain = _grow(sequence, dimension, rng, bias=2.0)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search
    if time.time() < seq_deadline - 1.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, seq_deadline - 1.0)
    
    # Phase 3: Simulated annealing if time permits
    if time.time() < seq_deadline - 0.2:
        best_chain, best_energy = _simulated_annealing(sequence, best_chain, rng, seq_deadline - 0.2)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
