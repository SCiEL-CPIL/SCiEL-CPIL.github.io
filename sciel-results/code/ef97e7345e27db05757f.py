import random
import time
import math

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 15.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 330.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Generate 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts."""
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
    """
    One randomized greedy chain growth.
    Heuristic: prefer sites that create new H-H contacts, then free space.
    This is the structural idea from the high-scoring inspiration.
    """
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score based on potential contacts and free space
            score = 0.0
            
            # If current residue is H, check for potential H-H contacts with existing H residues
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 10.0  # High weight for immediate contact gain
            
            # Prefer sites with more free neighbors to avoid dead ends (lower weight)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.5 * free
            
            # Add randomness
            score += rng.random()
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _monte_carlo_search(sequence, initial_chain, rng, time_limit):
    """
    Monte Carlo search with pivot moves (1-move and 2-move).
    This is the refinement machinery from the current program, but with robust pivot moves
    to avoid the bugs in the previous SA implementation.
    """
    n = len(sequence)
    if n < 5:
        return initial_chain, _energy(sequence, initial_chain)
    
    chain = list(initial_chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Cooling schedule for Metropolis acceptance
    T_start = 3.0
    T_end = 0.05
    cooling_rate = 0.999
    
    T = T_start
    start_time = time.time()
    
    while True:
        if time.time() > time_limit:
            break
        if T <= T_end:
            break
            
        # Choose move type: 1-move (flip) or 2-move (crankshaft)
        # 1-moves are cheaper, 2-moves help escape local minima
        if rng.random() < 0.6:
            move_type = 1
        else:
            move_type = 2
            
        if move_type == 1:
            # 1-move: Flip a single residue i.
            # Valid if i-1 and i+1 are at distance 2 from each other.
            # In that case, there is exactly one other valid position for i.
            i = rng.randrange(1, n - 1)
            p_prev = chain[i-1]
            p_next = chain[i+1]
            p_old = chain[i]
            
            # Check if p_prev and p_next are distance 2 apart
            dist = abs(p_prev[0]-p_next[0]) + abs(p_prev[1]-p_next[1]) + abs(p_prev[2]-p_next[2])
            if dist != 2:
                continue
            
            # Find the other common neighbor
            p_new = None
            for q in _neighbours(p_prev):
                if q == p_old:
                    continue
                if abs(q[0]-p_next[0]) + abs(q[1]-p_next[1]) + abs(q[2]-p_next[2]) == 1:
                    p_new = q
                    break
            
            if p_new is None or p_new in occupied:
                continue
            
            # Calculate energy delta
            delta_e = 0
            if sequence[i] == "H":
                # Lose contacts at p_old
                for q in _neighbours(p_old):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta_e += 1
                # Gain contacts at p_new
                for q in _neighbours(p_new):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta_e -= 1
            
            # Metropolis acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                occupied.pop(p_old)
                occupied[p_new] = i
                chain[i] = p_new
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        else:
            # 2-move: Crankshaft move.
            # Pick i. Move residues i and i+1.
            # i must be in [1, n-3] so that i-1 and i+2 exist.
            i = rng.randrange(1, n - 2)
            p_i_old = chain[i]
            p_ip1_old = chain[i+1]
            p_im1 = chain[i-1]
            p_ip2 = chain[i+2]
            
            # Generate possible new positions for i
            # Must be neighbor of p_im1, not p_i_old, not p_ip1_old, and not occupied
            candidates_i = []
            for q in _neighbours(p_im1):
                if q == p_i_old or q == p_ip1_old or q in occupied:
                    continue
                candidates_i.append(q)
            
            if not candidates_i:
                continue
            
            # For each candidate for i, generate candidates for i+1
            # Must be neighbor of new_i and p_ip2, not p_i_old, not p_ip1_old, and not occupied
            best_delta = float('inf')
            best_new_i = None
            best_new_ip1 = None
            
            for new_i in candidates_i:
                for new_ip1 in _neighbours(new_i):
                    if new_ip1 == p_i_old or new_ip1 == p_ip1_old or new_ip1 in occupied:
                        continue
                    # Check if new_ip1 is connected to p_ip2
                    if abs(new_ip1[0]-p_ip2[0]) + abs(new_ip1[1]-p_ip2[1]) + abs(new_ip1[2]-p_ip2[2]) != 1:
                        continue
                    
                    # Calculate delta energy
                    delta_e = 0
                    # Residue i
                    if sequence[i] == "H":
                        for q in _neighbours(p_i_old):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                                delta_e += 1
                        for q in _neighbours(new_i):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                                delta_e -= 1
                        # Check contact between i and i+1? No, they are adjacent.
                        # But check if new_i and new_ip1 form a contact? No, adjacent.
                        # However, we need to check if moving i creates a contact with i+1's old position?
                        # No, i+1 is moving too.
                        # We need to check if new_i is adjacent to new_ip1? Yes, by construction.
                        # We need to check if new_i is adjacent to any other residue? Handled by loop above.
                        # We need to check if new_ip1 is adjacent to any other residue? Handled below.
                    
                    # Residue i+1
                    if sequence[i+1] == "H":
                        for q in _neighbours(p_ip1_old):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                                delta_e += 1
                        for q in _neighbours(new_ip1):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                                delta_e -= 1
                    
                    # One subtlety: if sequence[i] == "H" and sequence[i+1] == "H",
                    # and they were not in contact before (which is true as they are adjacent),
                    # and they are not in contact after (also true as they are adjacent),
                    # then no change.
                    # But what if new_i is adjacent to new_ip1? It is.
                    # What if new_i is adjacent to p_ip1_old? No, p_ip1_old is being vacated.
                    # What if new_ip1 is adjacent to p_i_old? No, p_i_old is being vacated.
                    
                    # We must also check if new_i and new_ip1 create a contact between themselves?
                    # No, they are chain neighbors.
                    
                    # However, we must ensure that the move doesn't create a self-intersection
                    # with the rest of the chain. The `occupied` check handles this,
                    # but we must remember that p_i_old and p_ip1_old are still in `occupied`
                    # during the check. So `q in occupied` will be True for p_i_old and p_ip1_old.
                    # We explicitly excluded them.
                    
                    if delta_e < best_delta:
                        best_delta = delta_e
                        best_new_i = new_i
                        best_new_ip1 = new_ip1
            
            if best_new_i is not None:
                if best_delta < 0 or rng.random() < math.exp(-best_delta / T):
                    # Apply move
                    occupied.pop(p_i_old)
                    occupied.pop(p_ip1_old)
                    occupied[best_new_i] = i
                    occupied[best_new_ip1] = i + 1
                    chain[i] = best_new_i
                    chain[i+1] = best_new_ip1
                    current_energy += best_delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
        
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using multiple MC restarts.
    Combines the heuristic growth from the inspiration with the robust MC search from the current program.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Generate initial chains using heuristic growth
    # Number of growths depends on sequence length
    if n <= 48:
        num_growths = 50
    elif n <= 80:
        num_growths = 30
    else:
        num_growths = 20
    
    init_chains = []
    for _ in range(num_growths):
        if _remaining_time() < 10.0:
            break
        c = _grow(sequence, dimension, rng)
        if c:
            e = _energy(sequence, c)
            init_chains.append((e, c))
    
    if not init_chains:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Sort by energy
    init_chains.sort(key=lambda x: x[0])
    current_best_chain = init_chains[0][1]
    current_best_energy = init_chains[0][0]
    
    # Phase 2: Run MC search on the best initial chain
    # Allocate most of the remaining time to the best chain
    time_limit = time.time() + _remaining_time() - 5.0
    improved_chain, improved_energy = _monte_carlo_search(sequence, current_best_chain, rng, time_limit)
    
    if improved_energy < current_best_energy:
        current_best_chain = improved_chain
        current_best_energy = improved_energy
    
    # Phase 3: If time remains, run MC search on the 2nd best initial chain
    if _remaining_time() > 10.0 and len(init_chains) > 1:
        time_limit2 = time.time() + _remaining_time() - 2.0
        improved_chain2, improved_energy2 = _monte_carlo_search(sequence, init_chains[1][1], rng, time_limit2)
        if improved_energy2 < current_best_energy:
            current_best_chain = improved_chain2
            current_best_energy = improved_energy2
    
    return [list(p) for p in current_best_chain]
