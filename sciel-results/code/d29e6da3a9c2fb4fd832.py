"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate total energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with End, Reversal, and Segment moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Time management
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
    if not hasattr(fold, '_calls_made'):
        fold._calls_made = 0
    fold._calls_made += 1
    
    elapsed = time.time() - fold._start_time
    total_budget = 230.0
    remaining = total_budget - elapsed
    
    seqs_total = 26
    seqs_left = seqs_total - fold._calls_made + 1
    if seqs_left <= 0:
        time_per_seq = remaining
    else:
        time_per_seq = remaining / seqs_left
        
    max_time = max(0.1, time_per_seq)
    
    rng = random.Random(42 + len(sequence))
    
    def _grow_greedy(seq, dim, rng):
        start = (0,) * dim
        chain = [start]
        occupied = {start: 0}
        for i in range(1, len(seq)):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if seq[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied and occupied[q] < i - 1 and seq[occupied[q]] == "H":
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                options.append((gain + 0.5 * free + rng.random(), site))
            if not options:
                return None
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        return chain

    best_chain = None
    best_energy = 0
    
    for _ in range(10):
        c = _grow_greedy(sequence, dimension, rng)
        if c:
            e = _energy(sequence, c)
            if best_chain is None or e < best_energy:
                best_chain = c
                best_energy = e
    if best_chain is None:
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    current_chain = best_chain
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_sa = time.time()
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() - start_sa < max_time:
        if T < min_T:
            break
            
        move_type = rng.random()
        new_chain = None
        new_energy = None
        
        if move_type < 0.5:
            # End Move
            last_site = current_chain[-1]
            candidates = [ns for ns in _neighbours(last_site) if ns not in occupied]
            if candidates:
                ns = rng.choice(candidates)
                new_chain = current_chain[:-1] + [ns]
                new_energy = _energy(sequence, new_chain)
                
        elif move_type < 0.8:
            # Reversal
            if n > 2:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = current_chain[:i] + current_chain[i:j+1][::-1] + current_chain[j+1:]
                new_energy = _energy(sequence, new_chain)
                
        else:
            # Segment Move
            if n > 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 1, n - 1)
                # Pick a direction
                axis = rng.randint(0, dimension - 1)
                step = rng.choice([-1, 1])
                
                # Check if moving segment i..j by (axis, step) is valid
                # The segment moves as a rigid body.
                # We need to ensure new positions are not occupied by residues outside the segment.
                
                # Get set of occupied sites outside the segment
                outside_sites = set()
                for idx, site in enumerate(current_chain):
                    if idx < i or idx > j:
                        outside_sites.add(site)
                
                valid = True
                new_segment = []
                for idx in range(i, j + 1):
                    old_site = current_chain[idx]
                    new_site = list(old_site)
                    new_site[axis] += step
                    new_site = tuple(new_site)
                    new_segment.append(new_site)
                    if new_site in outside_sites:
                        valid = False
                        break
                    # Also check if new_site collides with other sites in the new segment?
                    # No, it's a rigid translation, so internal distances are preserved.
                    # But we must check if new_site is already in new_segment (from earlier idx)?
                    # No, translation is a bijection.
                    
                if valid:
                    new_chain = current_chain[:i] + new_segment + current_chain[j+1:]
                    new_energy = _energy(sequence, new_chain)

        if new_chain is None:
            T *= cooling
            continue
            
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            # Update occupied incrementally or rebuild?
            # Rebuild is O(N). Acceptable.
            occupied = {p: i for i, p in enumerate(current_chain)}
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        T *= cooling
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
