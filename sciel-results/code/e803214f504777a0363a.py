"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin
_PER_SEQ_BUDGET = _TOTAL_BUDGET / 26.0 * 0.92  # ~10.5s per sequence

# Precomputed neighbor offsets for 3D lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D lattice."""
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        yield (point[0] + dx, point[1] + dy, point[2] + dz)

def _neighbours_set(point):
    """Return set of 6 neighbors of a point."""
    return {(point[0] + dx, point[1] + dy, point[2] + dz) for dx, dy, dz in _NEIGHBOR_OFFSETS}

def _beam_grow(sequence, dimension, rng, beam_width=10, noise_scale=0.0):
    """
    Beam search chain growth.
    Maintains a beam of partial chains, expanding each and keeping the top beam_width.
    Score: accumulated H-H contacts (primary), free space (secondary), noise (diversity).
    """
    n = len(sequence)
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Initial state: single chain at origin
    # Each entry: (score, chain_tuple, occupied_set)
    start = (0, 0, 0)
    beam = [(0.0, (start,), {start: 0})]
    
    for i in range(1, n):
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Calculate new H-H contacts this site would make
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                # Free space: number of unoccupied neighbors
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                
                # Score: contacts dominate, free space breaks ties, noise for diversity
                noise = rng.random() * noise_scale
                new_score = score + gain * 10.0 + free * 0.1 + noise
                
                new_chain = chain + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            return None
        
        # Keep top beam_width entries
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    
    # Return the best complete chain
    best = max(beam, key=lambda x: x[0])
    return list(best[1])

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
                if abs(site[0] - next_pos[0]) + abs(site[1] - next_pos[1]) + abs(site[2] - next_pos[2]) == 1:
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
                    break  # Move to next i
    
    return best, best_e

def fold(sequence, dimension):
    """
    Best conformation using beam search growth + local search.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Time budget for this sequence
    current_time = time.time()
    elapsed_global = current_time - _TOTAL_START
    remaining_global = _TOTAL_BUDGET - elapsed_global
    
    # Allocate time for this sequence
    seq_deadline = min(current_time + _PER_SEQ_BUDGET, _TOTAL_START + _TOTAL_BUDGET)
    
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded)
    
    # Adaptive beam width and number of restarts
    if n <= 48:
        beam_width = 20
        n_restarts = 30
        noise_scales = [0.0, 0.1, 0.5, 1.0, 2.0]
    elif n <= 80:
        beam_width = 15
        n_restarts = 20
        noise_scales = [0.0, 0.1, 0.5, 1.0]
    else:
        beam_width = 10
        n_restarts = 12
        noise_scales = [0.0, 0.1, 0.5]
    
    # Phase 1: Beam search growths with varying noise for diversity
    for restart in range(n_restarts):
        if time.time() >= seq_deadline - 3.0:  # Leave 3s for local search
            break
        
        # Vary noise for diversity
        noise = noise_scales[restart % len(noise_scales)]
        # Also vary beam width slightly
        bw = beam_width + (restart % 3)  # Slight variation
        
        chain = _beam_grow(sequence, dimension, rng, beam_width=bw, noise_scale=noise)
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
    
    # Phase 2: Local search on the best chain
    if time.time() < seq_deadline - 0.5:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, seq_deadline - 0.5)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
