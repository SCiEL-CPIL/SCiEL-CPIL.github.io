"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute total H-H contact energy for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=8, max_iter=200):
    """
    Beam search: maintain multiple partial chains, extend each to the best
    scoring options, keep top beam_width by accumulated energy.
    """
    start = (0,) * dimension
    # Each beam entry: (chain, occupied_set, current_energy)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Score: immediate H-H contacts formed
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                # Slight preference for sites with more free space (avoids dead ends)
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain + 0.05 * free
                new_energy = energy - gain  # energy is negative
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beams.append((new_chain, new_occupied, new_energy, score))
        
        if not new_beams:
            break
        
        # Keep top beam_width by score (higher score = more contacts + free space)
        new_beams.sort(key=lambda x: -x[3])
        beams = [(b[0], b[1], b[2]) for b in new_beams[:beam_width]]
    
    if not beams:
        return None
    
    # Return the best complete chain by actual energy
    best_chain, best_energy = None, 1
    for chain, occupied, energy in beams:
        if len(chain) == len(sequence):
            actual = _compute_energy(sequence, chain)
            if actual < best_energy:
                best_chain, best_energy = chain, actual
    
    return best_chain


def _pivot_move(chain, i, dimension, rng):
    """
    Try a pivot move: rotate the sub-chain from position i to the end
    around chain[i-1]. Returns new chain or None if invalid.
    """
    if i <= 1 or i >= len(chain):
        return None
    
    prefix = chain[:i]
    suffix = chain[i:]
    
    # Get the point before the pivot
    pivot_point = prefix[-1]
    
    # Try all 6 possible directions for the new first point of suffix
    for axis in range(dimension):
        for step in (-1, 1):
            new_first = list(pivot_point)
            new_first[axis] += step
            new_first = tuple(new_first)
            
            if new_first in set(prefix[:-1]):
                continue
            
            # Build new suffix by translating
            offset = (new_first[0] - suffix[0][0],
                      new_first[1] - suffix[0][1],
                      new_first[2] - suffix[0][2])
            
            new_suffix = []
            valid = True
            for p in suffix:
                np = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                new_suffix.append(np)
            
            # Check self-avoidance with prefix
            new_full = prefix + new_suffix
            if len(set(new_full)) != len(new_full):
                continue
            
            # Check unit steps
            for k in range(len(new_full) - 1):
                a, b = new_full[k], new_full[k+1]
                dist = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
                if dist != 1:
                    valid = False
                    break
            
            if valid:
                return new_full
    
    return None


def _local_refine(sequence, chain, dimension, max_moves=50, rng=None):
    """Improve a chain using random pivot moves with energy-based acceptance."""
    if rng is None:
        rng = random.Random()
    
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    for _ in range(max_moves):
        # Pick a random pivot position
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move(current, i, dimension, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        
        # Accept if better, or with small probability if worse (simulated annealing lite)
        if new_energy < current_energy or rng.random() < 0.1:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
    
    return best, best_energy


def fold(sequence, dimension):
    """
    Fold using beam search for initial conformations, then local refinement.
    """
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Phase 1: Beam search with different beam widths
    beam_widths = [6, 10, 15]
    for bw in beam_widths:
        if time.time() > deadline - 30:
            break
        
        chain = _beam_search(sequence, dimension, beam_width=bw)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    # Phase 2: Local refinement on best chain
    if best_chain is not None and time.time() < deadline - 10:
        refined, refined_energy = _local_refine(sequence, best_chain, dimension, 
                                                  max_moves=100, rng=rng)
        if refined_energy < best_energy:
            best_chain = refined
            best_energy = refined_energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
