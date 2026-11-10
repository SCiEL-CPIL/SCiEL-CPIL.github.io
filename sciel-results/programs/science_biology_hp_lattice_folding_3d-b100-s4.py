"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 10.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate the energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _build_h_positions(sequence):
    """Precompute set of H residue indices for fast lookup."""
    return set(i for i, c in enumerate(sequence) if c == "H")

def _contact_delta(sequence, h_pos, i, old_site, new_site, occupied):
    """
    Compute energy delta for moving residue i from old_site to new_site.
    Returns: negative delta means energy improves (more contacts).
    """
    if i not in h_pos:
        return 0
    
    delta = 0
    # Lose contacts at old_site
    for q in _neighbours(old_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta += 1
    # Gain contacts at new_site
    for q in _neighbours(new_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta -= 1
    return delta

def _beam_search_fold(sequence, dimension, beam_width, rng, time_limit):
    """
    Beam search for HP lattice folding.
    Returns the best chain found or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
        
    start_point = (0,) * dimension
    # State: (score, energy, chain, occupied_dict, last_point)
    # score = energy + heuristic_penalty
    # We want to minimize score.
    
    # Initial state
    start_chain = [start_point]
    start_occupied = {start_point: 0}
    start_energy = 0
    # Heuristic: 0 for start
    start_score = 0
    
    beam = [(start_score, start_energy, start_chain, start_occupied, start_point)]
    
    h_pos = _build_h_positions(sequence)
    
    for i in range(1, n):
        res = sequence[i]
        new_beam_candidates = []
        
        # Time check inside loop for long chains
        if time.time() > time_limit:
            return None
            
        for score, energy_so_far, chain, occupied_dict, last_point in beam:
            # Try all neighbors of last_point
            for nb in _neighbours(last_point):
                if nb in occupied_dict:
                    continue
                
                # Calculate incremental energy
                # New contacts are H-H pairs where one is 'nb' and the other is in chain[:-1]
                # And they are not adjacent in the chain (index diff > 1)
                delta_e = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in occupied_dict:
                            j = occupied_dict[q]
                            # j is the index of the neighbor in the chain
                            # Check if j is not the previous residue (i-1)
                            # Since we are adding residue i, the previous is i-1.
                            # If j == i-1, it's a backbone bond, not a contact.
                            if j != i - 1:
                                if sequence[j] == "H":
                                    delta_e += 1
                new_energy = energy_so_far - delta_e
                
                # Heuristic: Compactness
                # Count free neighbors of nb. Lower free count is better (more compact).
                # Penalty: 0.1 * free_count
                free_count = 0
                for q in _neighbours(nb):
                    if q not in occupied_dict:
                        free_count += 1
                
                # Score: Lower is better
                # Energy is negative. We want low energy (more negative).
                # Free count is positive. We want low free count.
                # score = new_energy + 0.1 * free_count
                # Example: Energy -10, Free 2 -> -9.8
                # Example: Energy -10, Free 5 -> -9.5
                # -9.8 < -9.5, so we prefer lower free count. Correct.
                
                # Add small random noise to break ties and encourage diversity
                noise = rng.random() * 0.01
                heuristic_weight = 0.1
                new_score = new_energy + heuristic_weight * free_count + noise
                
                # Prepare new state
                # Copy chain and occupied dict
                # Optimization: Instead of copying the whole list/dict every time, 
                # we could use persistent data structures, but for N~136, copy is okay.
                new_chain = chain + [nb]
                new_occupied = occupied_dict.copy()
                new_occupied[nb] = i
                
                new_beam_candidates.append((new_score, new_energy, new_chain, new_occupied, nb))
        
        if not new_beam_candidates:
            return None
            
        # Sort by score and keep top beam_width
        # Using a heap might be faster if beam_width is small relative to candidates,
        # but sorting is simple. Candidates = beam_width * 6 approx.
        # For beam_width=100, candidates=600. Sorting 600 items is fast.
        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[:beam_width]
        
        # If beam is empty, fail
        if not beam:
            return None
            
    # Find the best energy in the final beam
    best_state = min(beam, key=lambda x: x[1])
    return best_state[2]

def _local_search(sequence, chain, max_iters, rng, time_limit, use_annealing=True, polish=False):
    """
    Local search with optional simulated annealing and optional polish mode.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    h_pos = _build_h_positions(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Annealing parameters
    if use_annealing:
        if polish:
            T_start = 0.5
            T_end = 0.01
        else:
            T_start = 2.0
            T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    iters = 0
    no_improve_count = 0
    patience = 200 if polish else 500
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        # Pick a random interior residue to try moving
        i = rng.randrange(1, n - 1)
        old_site = chain[i]
        
        improved = False
        # Try each neighbor of old_site as a new position
        for new_site in _neighbours(old_site):
            # Check if new_site is free (not occupied by any other residue)
            if new_site in occupied:
                continue
            
            # Check adjacency with neighbors
            if i > 0:
                prev = chain[i-1]
                if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                    continue
            if i < n - 1:
                nxt = chain[i+1]
                if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                    continue
            
            # Compute energy change using fast contact delta
            delta = _contact_delta(sequence, h_pos, i, old_site, new_site, occupied)
            
            new_energy = current_energy + delta
            
            # Acceptance criterion
            accept = False
            if new_energy < current_energy:
                accept = True
            elif use_annealing and iters < max_iters:
                T = T_start * (cooling_rate ** iters)
                if T > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Apply the move
                occupied.pop(old_site)
                occupied[new_site] = i
                chain[i] = new_site
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True
                break  # Only one move per iteration
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search for construction and SA for refinement.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = float('inf')
    
    n = len(sequence)
    
    # Determine beam width based on chain length
    if n <= 48:
        beam_width = 200
        num_bs_runs = 5
    elif n <= 80:
        beam_width = 150
        num_bs_runs = 3
    else:
        beam_width = 100
        num_bs_runs = 2
        
    # Time allocation:
    # 50% for Beam Search
    # 50% for Local Search (SA)
    
    total_time_available = _remaining_time()
    bs_time_budget = total_time_available * 0.5
    ls_time_budget = total_time_available * 0.5
    
    bs_deadline = time.time() + bs_time_budget
    
    # Run Beam Search
    for run in range(num_bs_runs):
        if time.time() > bs_deadline:
            break
            
        # Adjust beam width for subsequent runs if time is tight
        current_bw = beam_width
        if run > 0 and time.time() > bs_deadline - 5.0:
             break
             
        chain = _beam_search_fold(sequence, dimension, current_bw, rng, bs_deadline)
        if chain is None:
            continue
            
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    # Fallback if Beam Search failed
    if best_chain is None:
        # Simple greedy growth fallback
        start = (0,) * dimension
        chain, occupied = [start], {start: 0}
        for i in range(1, len(sequence)):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                options.append(site)
            if not options:
                break
            site = rng.choice(options)
            occupied[site] = i
            chain.append(site)
        if len(chain) == len(sequence):
            best_chain = chain
            best_energy = _energy(sequence, chain)
        else:
            # Absolute fallback: straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
            best_energy = _energy(sequence, best_chain)
            
    # Local Search Refinement
    if _remaining_time() > 5.0:
        ls_deadline = time.time() + ls_time_budget
        
        # Run SA on the best chain
        # Use a generous number of iterations, limited by time
        max_ls_iters = max(1000, min(20000, 100 * len(sequence)))
        
        improved_chain = _local_search(
            sequence, 
            best_chain, 
            max_ls_iters, 
            rng, 
            ls_deadline, 
            use_annealing=True
        )
        
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
        # If time remains, do a polish pass
        if _remaining_time() > 3.0:
            polish_deadline = time.time() + _remaining_time() * 0.8
            max_polish_iters = max(1000, min(20000, 100 * len(sequence)))
            
            polished_chain = _local_search(
                sequence, 
                best_chain, 
                max_polish_iters, 
                rng, 
                polish_deadline, 
                use_annealing=True, 
                polish=True
            )
            
            polished_energy = _energy(sequence, polished_chain)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
