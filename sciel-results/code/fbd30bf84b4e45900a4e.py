"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

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

def _grow(sequence, dimension, rng):
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
            # Slightly more random to ensure diversity
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Local search with single-residue moves, 2-residue cranks, and 3-residue cranks.
    Uses simulated annealing-style acceptance to escape local minima.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    total_iters = max_iters
    # Annealing parameters
    initial_temp = 3.0
    cooling_rate = 0.999
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        improved = False
        temperature = initial_temp * (cooling_rate ** iters)
        
        # Choose move type: 0=single, 1=2-crank, 2=3-crank
        # Probabilities: 0.35, 0.35, 0.30
        r = rng.random()
        if r < 0.35:
            move_type = 0
        elif r < 0.70:
            move_type = 1
        else:
            move_type = 2

        if move_type == 0:
            # Single residue move
            if n < 3:
                continue
            i = rng.randrange(1, n - 1)
            old_site = chain[i]
            
            for new_site in _neighbours(old_site):
                if new_site in occupied:
                    if occupied[new_site] != i:
                        continue
                    else:
                        continue
                
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                        continue
                if i < n - 1:
                    nxt = chain[i+1]
                    if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                        continue
                
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours(old_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta += 1
                    for q in _neighbours(new_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta -= 1
                
                new_energy = current_energy + delta
                # Simulated annealing acceptance
                if delta < 0 or (temperature > 0.01 and rng.random() < pow(2.718281828, -(-delta) / temperature)):
                    occupied.pop(old_site)
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    improved = True
                    break
        
        elif move_type == 1:
            # 2-residue crank
            if n < 4:
                continue
            i = rng.randrange(1, n - 2)
            j = i + 1
            old_i = chain[i]
            old_j = chain[j]
            
            for new_i in _neighbours(old_i):
                if new_i in occupied:
                    occ = occupied[new_i]
                    if occ != i and occ != j:
                        continue
                else:
                    pass
                
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                        continue
                
                for new_j in _neighbours(new_i):
                    if new_j in occupied:
                        occ = occupied[new_j]
                        if occ != i and occ != j:
                            continue
                    else:
                        pass
                    
                    if j < n - 1:
                        nxt = chain[j+1]
                        if abs(new_j[0]-nxt[0]) + abs(new_j[1]-nxt[1]) + abs(new_j[2]-nxt[2]) != 1:
                            continue
                    
                    if new_i == old_j or new_j == old_i:
                        continue
                    
                    delta = 0
                    if sequence[i] == "H":
                        for q in _neighbours(old_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(i-k) > 1:
                                delta += 1
                        for q in _neighbours(new_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(i-k) > 1:
                                delta -= 1
                    if sequence[j] == "H":
                        for q in _neighbours(old_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(j-k) > 1:
                                delta += 1
                        for q in _neighbours(new_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(j-k) > 1:
                                delta -= 1
                    
                    new_energy = current_energy + delta
                    if delta < 0 or (temperature > 0.01 and rng.random() < pow(2.718281828, -(-delta) / temperature)):
                        occupied.pop(old_i)
                        occupied.pop(old_j)
                        occupied[new_i] = i
                        occupied[new_j] = j
                        chain[i] = new_i
                        chain[j] = new_j
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                        improved = True
                        break
                if improved:
                    break
        
        else:
            # 3-residue crank: pivot at i+1, move i to new position
            if n < 5:
                continue
            i = rng.randrange(1, n - 3)
            # Move residue i, keeping i-1, i+1, i+2 fixed
            # i must move to a neighbor of old_i that is adjacent to chain[i-1]
            # and such that chain[i+1] is still adjacent to new_i
            # Actually, let's do a proper 3-crank: rotate the triad (i, i+1, i+2)
            # Pick i. The segment is chain[i], chain[i+1], chain[i+2].
            # We want to move all three. A simpler approach: 
            # Try to move chain[i+2] to a new position, but also allow chain[i+1] to shift.
            # Let's do: pick i. Try new position for chain[i] (adj to chain[i-1]),
            # then chain[i+1] must be adj to new chain[i], then chain[i+2] must be adj to new chain[i+1].
            # This is a 3-step walk.
            
            old_i = chain[i]
            old_i1 = chain[i+1]
            old_i2 = chain[i+2]
            
            found_move = False
            for new_i in _neighbours(old_i):
                if new_i in occupied:
                    occ = occupied[new_i]
                    if occ != i and occ != i+1 and occ != i+2:
                        continue
                else:
                    pass
                
                # Check adjacency to chain[i-1]
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                        continue
                
                # Now try new positions for i+1
                for new_i1 in _neighbours(new_i):
                    if new_i1 in occupied:
                        occ = occupied[new_i1]
                        if occ != i and occ != i+1 and occ != i+2:
                            continue
                    else:
                        pass
                    
                    if new_i1 == old_i:
                        continue
                    
                    # Check adjacency to new_i (already ensured by _neighbours)
                    # Now try new positions for i+2
                    for new_i2 in _neighbours(new_i1):
                        if new_i2 in occupied:
                            occ = occupied[new_i2]
                            if occ != i and occ != i+1 and occ != i+2:
                                continue
                        else:
                            pass
                        
                        if new_i2 == old_i or new_i2 == old_i1:
                            continue
                        
                        # Check adjacency to chain[i+3]
                        if i+2 < n - 1:
                            nxt = chain[i+3]
                            if abs(new_i2[0]-nxt[0]) + abs(new_i2[1]-nxt[1]) + abs(new_i2[2]-nxt[2]) != 1:
                                continue
                        
                        # Compute energy delta for residues i, i+1, i+2
                        delta = 0
                        for idx, old_pos, new_pos in [(i, old_i, new_i), (i+1, old_i1, new_i1), (i+2, old_i2, new_i2)]:
                            if sequence[idx] == "H":
                                for q in _neighbours(old_pos):
                                    k = occupied.get(q)
                                    if k is not None and k != i and k != i+1 and k != i+2 and sequence[k] == "H" and abs(idx-k) > 1:
                                        delta += 1
                                for q in _neighbours(new_pos):
                                    k = occupied.get(q)
                                    if k is not None and k != i and k != i+1 and k != i+2 and sequence[k] == "H" and abs(idx-k) > 1:
                                        delta -= 1
                        
                        new_energy = current_energy + delta
                        if delta < 0 or (temperature > 0.01 and rng.random() < pow(2.718281828, -(-delta) / temperature)):
                            occupied.pop(old_i)
                            occupied.pop(old_i1)
                            occupied.pop(old_i2)
                            occupied[new_i] = i
                            occupied[new_i1] = i+1
                            occupied[new_i2] = i+2
                            chain[i] = new_i
                            chain[i+1] = new_i1
                            chain[i+2] = new_i2
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_chain = list(chain)
                            improved = True
                            found_move = True
                            break
                    if found_move:
                        break
                if found_move:
                    break

        if not improved and iters % 200 == 0:
            if time.time() > time_limit:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with optional local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    if n <= 48:
        num_growths = 300
    elif n <= 80:
        num_growths = 200
    else:
        num_growths = 150
    
    candidate_chains = []
    for attempt in range(num_growths):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
        candidate_chains.append((value, chain))
    
    candidate_chains.sort(key=lambda x: x[0])
    
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        num_ls_runs = min(len(candidate_chains), max(3, int(remaining / 12)))
        per_run_time = (remaining - 5.0) / num_ls_runs
        
        for idx in range(num_ls_runs):
            if _remaining_time() < 5.0:
                break
            if idx == 0:
                start_chain = candidate_chains[0][1]
            elif idx == 1:
                start_chain = candidate_chains[min(1, len(candidate_chains)-1)][1]
            else:
                pool = candidate_chains[min(idx, len(candidate_chains)-1):]
                if pool:
                    start_chain = pool[0][1]
                else:
                    start_chain = candidate_chains[0][1]
            
            ls_time_limit = time.time() + per_run_time
            max_ls_iters = max(1000, min(30000, 150 * len(sequence)))
            improved_chain = _local_search(sequence, list(start_chain), max_ls_iters, rng, ls_time_limit)
            improved_energy = _energy(sequence, improved_chain)
            if improved_energy < best_energy:
                best, best_energy = improved_chain, improved_energy
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
