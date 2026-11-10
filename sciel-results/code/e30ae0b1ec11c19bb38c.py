"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last_pos = chain[-1]
        # Pre-compute neighbors of the last position
        neighbors = _neighbours2d(last_pos)
        
        for site in neighbors:
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts
            gain = 0
            if sequence[i] == "H":
                # Check neighbors of the candidate site
                for q in _neighbours2d(site):
                    if q in occupied:
                        j = occupied[q]
                        # Must not be adjacent in the chain (j != i-1)
                        # And must be H
                        if j != i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Calculate free space: number of empty neighbors
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied:
                    free += 1
            
            # Heuristic: Strongly prefer gain, moderately prefer free space
            # Add some randomness to allow exploration
            score = gain * 4.0 + free * 0.5 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Lower is better (more negative)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == "H":
            p = chain[i]
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, occupied, i, old_pos, new_pos):
    """
    Calculate the change in energy when moving residue i from old_pos to new_pos.
    Assumes occupied reflects the state where i is at old_pos.
    """
    delta = 0
    if sequence[i] == 'H':
        # Remove old contacts
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1 # Energy increases (becomes less negative)
        
        # Add new contacts
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1 # Energy decreases (becomes more negative)
                
    return delta

def _is_valid_move(chain, occupied, i, new_pos):
    """Check if moving residue i to new_pos is valid (self-avoiding)."""
    if new_pos in occupied:
        return False
    # Check if the new position would cause an adjacency violation
    # The only adjacency allowed is with i-1 and i+1
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None:
            if abs(j - i) > 1:
                return False
    return True

def _local_search(sequence, chain, max_iter=2000, rng=None):
    """Local search with single-residue moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    no_improve_count = 0
    max_no_improve = 100
    
    for _ in range(max_iter):
        if no_improve_count > max_no_improve:
            break
            
        i = rng.randint(1, n - 2)
        current_pos = best_chain[i]
        
        # Try all 4 neighbors
        for new_pos in _neighbours2d(current_pos):
            if not _is_valid_move(best_chain, occupied, i, new_pos):
                continue
            
            delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
            new_energy = best_energy + delta
            
            # Accept if it improves, or with some probability if it's neutral
            if delta < 0 or (delta == 0 and rng.random() < 0.1):
                # Make the move
                best_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                best_energy = new_energy
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        if no_improve_count > max_no_improve:
            break

    return best_chain, best_energy

def _simulated_annealing(sequence, chain, max_iter=10000, initial_temp=5.0, final_temp=0.01, rng=None, deadline=None):
    """Simulated Annealing with single-residue moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, max_iter))
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        temp = initial_temp * (cooling_rate ** iteration)
        if temp < 0.001:
            break
        
        i = rng.randint(1, n - 2)
        current_pos = current_chain[i]
        
        # Try a random neighbor
        new_pos = rng.choice(list(_neighbours2d(current_pos)))
        
        if not _is_valid_move(current_chain, occupied, i, new_pos):
            continue
            
        delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
        new_energy = current_energy + delta
        
        # Accept if it improves, or with probability exp(-delta/temp)
        if delta < 0 or (temp > 0.001 and rng.random() < math.exp(-delta / temp)):
            current_chain[i] = new_pos
            occupied.pop(current_pos)
            occupied[new_pos] = i
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using randomized greedy growths, local search, and SA."""
    start_time = time.time()
    # Safety margin: The evaluator kills at 210s. We stop at 205s to be safe.
    deadline = start_time + 205.0
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Adjust parameters based on sequence length
    if n <= 30:
        num_restarts = 1000
        ls_iter = 1000
        sa_iter = 5000
    elif n <= 60:
        num_restarts = 500
        ls_iter = 500
        sa_iter = 2000
    else:
        num_restarts = 200
        ls_iter = 200
        sa_iter = 1000
        
    # Phase 1: Multiple restarts with local search
    for attempt in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        chain, energy = _local_search(sequence, chain, max_iter=ls_iter, rng=rng)
        
        if energy < best_energy:
            best, best_energy = chain, energy
            
    # Phase 2: Simulated Annealing on the best solution
    if best is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        # Estimate iterations per second
        est_iter_per_sec = 500 if n < 40 else 200
        sa_iter_est = int(remaining_time * est_iter_per_sec)
        sa_iter = max(1000, min(sa_iter_est, 50000))
        
        best, best_energy = _simulated_annealing(
            sequence, best, max_iter=sa_iter, initial_temp=5.0, final_temp=0.01, rng=rng,
            deadline=deadline - 2.0
        )
    
    # Phase 3: Final local search polish
    if best is not None and time.time() < deadline - 1.0:
        best, best_energy = _local_search(sequence, best, max_iter=2000, rng=rng)
    
    if best is None:
        # Fallback: straight line (invalid for HP but valid as a walk)
        best = [(i, 0) for i in range(len(sequence))]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
