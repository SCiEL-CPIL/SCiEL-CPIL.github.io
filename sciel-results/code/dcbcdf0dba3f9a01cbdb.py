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
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _delta_energy(sequence, chain, where, occupied, i, j, old_pos, new_pos):
    """Compute energy change when moving residue j from old_pos to new_pos.
    
    We track which contacts are broken and formed.
    A contact exists between residues a and b if they are H, not adjacent in chain (|a-b|>1),
    and their lattice positions are neighbours.
    """
    if sequence[j] != "H":
        # Moving a P residue doesn't change H-H contacts directly,
        # but it might block or unblock future moves... actually no,
        # energy only depends on H-H contacts. Moving P doesn't change energy.
        return 0
    
    # Contacts broken: j was at old_pos, check neighbours of old_pos
    # For each neighbour site of old_pos, if there's a residue k there that is H
    # and |j-k| > 1, then we had a contact. Now we remove it.
    delta = 0
    
    # Broken contacts
    for q in _neighbours(old_pos):
        k = where.get(q)
        if k is not None and k != j and abs(j - k) > 1 and sequence[k] == "H":
            delta += 1  # we lose one contact
    
    # Formed contacts
    for q in _neighbours(new_pos):
        k = where.get(q)
        if k is not None and k != j and abs(j - k) > 1 and sequence[k] == "H":
            delta -= 1  # we gain one contact
    
    return delta


def _simulated_annealing(sequence, dimension, initial_chain, t0, cooling, max_iters, rng, deadline):
    """Improve a chain using simulated annealing with pivot/flip moves."""
    chain = list(initial_chain)
    where = {p: i for i, p in enumerate(chain)}
    occupied = set(chain)
    
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    if n < 4:
        return best_chain
    
    # Precompute H positions for faster checking
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    t = t0
    iter_count = 0
    no_improve = 0
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        
        # Choose a move: pick a random residue j (not 0 or n-1) and try to move it
        j = rng.randint(1, n - 2)
        
        # Find valid neighbours of current position of j
        old_pos = chain[j]
        candidates = []
        for q in _neighbours(old_pos):
            if q in occupied and q != old_pos:
                # q is occupied by another residue, can't move there
                continue
            if q in occupied:
                continue
            # Check if moving j to q is valid (self-avoiding)
            # q must not be occupied by any other residue
            if q not in occupied:
                candidates.append(q)
        
        if not candidates:
            iter_count += 1
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute energy change
        delta = _delta_energy(sequence, chain, where, occupied, j, j, old_pos, new_pos)
        
        # Acceptance criterion
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            # Accept the move
            # Update data structures
            occupied.remove(old_pos)
            occupied.add(new_pos)
            where.pop(old_pos)
            where[new_pos] = j
            chain[j] = new_pos
            
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        
        iter_count += 1
        t *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing with time budget."""
    start_time = time.time()
    total_deadline = start_time + 230  # Leave 10s safety margin for 240s budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate multiple greedy chains
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 means no valid chain yet
    
    # Number of greedy restarts based on chain length
    if n <= 50:
        num_greedy = 30
    elif n <= 80:
        num_greedy = 15
    else:
        num_greedy = 8
    
    for _ in range(num_greedy):
        if time.time() > total_deadline:
            break
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing on the best chain
    if n >= 4 and time.time() < total_deadline:
        # Set initial temperature based on best_energy
        # Energy is negative, so use absolute value
        t0 = max(1.0, abs(best_energy) * 0.5)
        cooling = 0.999
        remaining_time = total_deadline - time.time()
        
        if remaining_time > 5:
            improved = _simulated_annealing(
                sequence, dimension, best_chain, t0, cooling,
                int(remaining_time * 50), rng, total_deadline
            )
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy
    
    # Phase 3: If time remains, do additional SA restarts from perturbed best
    if n >= 4 and time.time() < total_deadline - 10:
        # Try a few more SA runs from perturbed versions of best
        for restart in range(3):
            if time.time() > total_deadline - 5:
                break
            
            # Perturb the best chain
            perturbed = list(best_chain)
            where = {p: i for i, p in enumerate(perturbed)}
            occupied = set(perturbed)
            
            # Move a random internal residue to a random valid neighbour
            for attempt in range(50):
                j = rng.randint(1, n - 2)
                old_pos = perturbed[j]
                candidates = []
                for q in _neighbours(old_pos):
                    if q not in occupied:
                        candidates.append(q)
                if candidates:
                    new_pos = rng.choice(candidates)
                    occupied.remove(old_pos)
                    occupied.add(new_pos)
                    where.pop(old_pos)
                    where[new_pos] = j
                    perturbed[j] = new_pos
                    break
            
            perturbed_energy = _energy(sequence, perturbed)
            
            if perturbed_energy < best_energy:
                best_chain = perturbed
                best_energy = perturbed_energy
            
            # SA on perturbed
            remaining = total_deadline - time.time()
            if remaining > 5:
                t0 = max(1.0, abs(best_energy) * 0.3)
                improved = _simulated_annealing(
                    sequence, dimension, best_chain, t0, 0.999,
                    int(remaining * 30), rng, total_deadline
                )
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_chain = improved
                    best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
