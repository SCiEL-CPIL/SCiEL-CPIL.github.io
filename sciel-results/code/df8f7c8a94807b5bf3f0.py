"""
HP lattice protein folding combining Beam Search for initial conformation
and Segment Relocation Local Search for refinement.
"""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 15s buffer -> 195s usable
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Calculate energy of a full chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_beam_search(sequence, dimension, beam_width=100, time_limit=5.0, rng=None):
    """
    Perform Beam Search for the given sequence.
    Returns the best full conformation found, or None if failed.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
        
    is_h = [c == 'H' for c in sequence]
    
    # State: (energy, chain, occupied_set, h_positions_set)
    # Using sets for O(1) lookup of occupied positions and H positions
    
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h_pos = {start_pos} if is_h[0] else set()
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h_pos)]
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            # prev_pos = chain[-2] if len(chain) > 1 else None # Not strictly needed for energy calc if we check all neighbors
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    # Check for new contacts
                    for q in _neighbours(nb):
                        if q in h_pos:
                            # Ensure q is not the previous residue (which is already in chain and h_pos)
                            # Actually, if q is in h_pos, it's a previous H.
                            # We need to ensure we don't count the bond to the previous residue if it was H.
                            # The bond to previous residue is (last_pos, nb).
                            # If last_pos is H, it is in h_pos.
                            # So if nb is H, and last_pos is H, we will see last_pos in _neighbours(nb) and it is in h_pos.
                            # This would count a "contact" which is actually a bond.
                            # We must exclude the previous residue.
                            if q != chain[-2]:
                                new_energy -= 1
                else:
                    new_h_pos = h_pos

                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        if not new_beam:
            return None
            
        # Prune
        if len(new_beam) > beam_width:
            # Sort by energy (ascending, so most negative is first)
            new_beam.sort(key=lambda x: x[0])
            
            # Keep top beam_width
            # To add diversity, we can randomly select from ties
            cutoff_energy = new_beam[beam_width - 1][0]
            
            keep = []
            maybe = []
            for state in new_beam:
                if state[0] < cutoff_energy:
                    keep.append(state)
                elif state[0] == cutoff_energy:
                    maybe.append(state)
                else:
                    break
            
            slots_needed = beam_width - len(keep)
            if slots_needed > 0:
                if rng:
                    rng.shuffle(maybe)
                else:
                    random.shuffle(maybe)
                keep.extend(maybe[:slots_needed])
            
            beam = keep
        else:
            beam = new_beam
            
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Pick the best state from the final beam
    best_state = min(beam, key=lambda x: x[0])
    chain = best_state[1]
    occupied = best_state[2]
    h_pos = best_state[3]
    
    # If we broke early due to time, extend greedily to ensure full length
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            prev_pos = chain[-2] if len(chain) > 1 else None
            
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != prev_pos:
                            e -= 1
                candidates.append((e, nb))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_e_val = candidates[0][0]
                best_opts = [nb for e, nb in candidates if e == best_e_val]
                if rng:
                    best_nb = rng.choice(best_opts)
                else:
                    best_nb = random.choice(best_opts)
                    
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
            else:
                # Dead end
                return None
                
    return chain

def _try_relocate_segment(sequence, chain, start_idx, end_idx, occupied_by_others, rng):
    """
    Attempt to relocate the segment chain[start_idx:end_idx+1] to a new position.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
    if not start_candidates:
        return None
    
    rng.shuffle(start_candidates)
    
    max_walks_per_start = 5
    
    for start_site in start_candidates:
        for _walk_attempt in range(max_walks_per_start):
            path = [start_site]
            visited = {start_site}
            
            for step in range(seg_len - 1):
                current = path[-1]
                options = []
                
                for site in _neighbours(current):
                    if site in visited or site in occupied_by_others:
                        continue
                    
                    dist_to_end = abs(site[0] - right_neighbor[0]) + abs(site[1] - right_neighbor[1])
                    if rng.random() < 0.3:
                        score = rng.random()
                    else:
                        score = dist_to_end + rng.random() * 2.0
                    options.append((score, site))
                
                if not options:
                    break
                
                options.sort(key=lambda x: x[0])
                next_site = options[0][1]
                path.append(next_site)
                visited.add(next_site)
            
            if len(path) == seg_len:
                last = path[-1]
                if abs(last[0] - right_neighbor[0]) + abs(last[1] - right_neighbor[1]) == 1:
                    return path
    
    return None

def _segment_move(sequence, chain, rng, max_attempts=200):
    """
    Local search via segment relocation moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_attempts):
        if _time_remaining() < 2.0:
            break
        
        max_seg_len = min(15, n - 2)
        seg_len = min(rng.randint(3, max_seg_len), n - 2)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        occupied_by_others = {p: i for i, p in enumerate(chain) if i < start_idx or i > end_idx}
        
        new_segment = _try_relocate_segment(sequence, chain, start_idx, end_idx, occupied_by_others, rng)
        
        if new_segment is None:
            continue
        
        new_chain = list(chain)
        for i, p in enumerate(new_segment):
            new_chain[start_idx + i] = p
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            # Quick follow-up
            extra_attempts = min(5, max_attempts - _ - 1)
            for _extra in range(extra_attempts):
                if _time_remaining() < 2.0:
                    break
                seg_len2 = min(rng.randint(3, min(10, n - 2)), n - 2)
                start_idx2 = rng.randint(1, n - seg_len2 - 1)
                end_idx2 = start_idx2 + seg_len2 - 1
                occupied_by_others2 = {p: i for i, p in enumerate(best_chain) if i < start_idx2 or i > end_idx2}
                new_segment2 = _try_relocate_segment(sequence, best_chain, start_idx2, end_idx2, occupied_by_others2, rng)
                if new_segment2 is None:
                    continue
                new_chain2 = list(best_chain)
                for i, p in enumerate(new_segment2):
                    new_chain2[start_idx2 + i] = p
                new_energy2 = _energy(sequence, new_chain2)
                if new_energy2 < best_energy:
                    best_chain = new_chain2
                    best_energy = new_energy2
    
    return best_chain

def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam Search
    # Allocate ~70% of time to Beam Search
    beam_phase_deadline = _time_remaining() * 0.70
    
    # Determine beam width based on sequence length
    if n <= 20:
        base_beam_width = 300
    elif n <= 50:
        base_beam_width = 150
    else:
        base_beam_width = 80
        
    # Run Beam Search multiple times with different seeds
    while _time_remaining() > beam_phase_deadline:
        # Estimate time per beam search run
        # For n=100, a beam search might take 1-2 seconds depending on width
        # We will just run it with a time limit that fits in the remaining window
        # To be safe, we set a small time limit per run if we have many runs left, 
        # or a larger one if we have few.
        
        # Simple heuristic: if we have > 20s left in beam phase, allow 5s per run.
        # If we have < 20s left, allow 2s per run.
        run_time_limit = 5.0 if _time_remaining() - beam_phase_deadline > 20.0 else 2.0
        
        # Adjust beam width slightly for diversity? 
        # Let's stick to base_beam_width but vary seed.
        
        chain = _fold_beam_search(sequence, dimension, base_beam_width, run_time_limit, rng)
        
        if chain is not None and len(chain) == n:
            # Validate
            occupied = set()
            valid = True
            for i, p in enumerate(chain):
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
                if i > 0:
                    prev = chain[i-1]
                    if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                        valid = False
                        break
            
            if valid:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
    
    if best_chain is None:
        # Fallback: simple linear chain
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local Search Refinement
    if _time_remaining() > 3.0:
        # Use remaining time for local search
        # Estimate attempts: each attempt takes ~0.001-0.01s? 
        # Let's just run a loop that checks time.
        max_local_attempts = min(2000, int(_time_remaining() * 50))
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=max_local_attempts)
        # Update best_energy just in case, though _segment_move returns improved chain
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
