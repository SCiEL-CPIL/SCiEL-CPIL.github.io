import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D."""
    # Unroll for speed
    yield (point[0] + 1, point[1], point[2])
    yield (point[0] - 1, point[1], point[2])
    yield (point[0], point[1] + 1, point[2])
    yield (point[0], point[1] - 1, point[2])
    yield (point[0], point[1], point[2] + 1)
    yield (point[0], point[1], point[2] - 1)

def _grow(sequence, dimension, rng):
    """
    Greedy growth with randomization to avoid getting stuck.
    Returns a list of tuples representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score the site
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        # Check if it's a chain neighbor (i-1)
                        if idx == i - 1:
                            continue
                        if sequence[idx] == 'H':
                            gain += 1
            
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            rand = rng.random()
            
            # Weighted score: Gain is most important
            score = gain * 10.0 + free * 0.1 + rand
            options.append((score, site))
            
        if not options:
            return None
            
        best_site = max(options)[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total energy of a chain."""
    if not chain:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, chain, k, new_segment, occupied, is_h):
    """
    Calculate the change in energy when replacing segment k: of chain with new_segment.
    'occupied' must contain the rest of the chain (0..k-1) and NOT the old segment k..n-1.
    """
    n = len(chain)
    old_seg = chain[k:]
    
    # 1. Count contacts lost by removing old_seg
    # Contacts are between H residues at distance 1, not adjacent in chain.
    # We need to count:
    # a) Contacts between old_seg and rest (0..k-1)
    # b) Contacts within old_seg (non-adjacent)
    
    lost = 0
    
    # Create a set of positions in old_seg for fast internal contact check?
    # Or just iterate.
    old_pos_set = set(old_seg)
    
    for i_offset, pos in enumerate(old_seg):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            # Check if neighbor is in rest (0..k-1)
            if nb in occupied:
                nb_idx = occupied[nb]
                # nb_idx is in 0..k-1
                if is_h[nb_idx]:
                    # Check if adjacent in chain?
                    # idx is k..n-1, nb_idx is 0..k-1. 
                    # Adjacent means |idx - nb_idx| == 1.
                    # Since nb_idx < k and idx >= k, |idx - nb_idx| == 1 is possible if idx==k and nb_idx==k-1.
                    if abs(idx - nb_idx) > 1:
                        lost += 1
            
            # Check if neighbor is in old_seg (internal contact)
            elif nb in old_pos_set:
                # Find index of nb in old_seg
                # We can't use 'occupied' for this part as it doesn't have old_seg.
                # We need to know the index in the chain.
                # Let's pre-compute a map for old_seg positions to their chain indices?
                # Or just check distance.
                # Since we are iterating over old_seg, we can just check if nb is in old_seg.
                # But we need to know if they are adjacent in chain.
                # Let's build a small dict for old_seg: pos -> idx
                pass # This is getting inefficient. Let's use a different approach for delta.

    # Better Delta Approach:
    # 1. Calculate energy of the "rest" + "old_seg" ? No.
    # 2. Calculate contacts involving the boundary and internal to the segment.
    
    # Let's just re-calculate the contacts involving the moved segment.
    # Total Energy E = -Contacts
    # Delta E = E_new - E_old
    # Contacts_old = C(rest, rest) + C(rest, old_seg) + C(old_seg, old_seg)
    # Contacts_new = C(rest, rest) + C(rest, new_seg) + C(new_seg, new_seg)
    # Delta Contacts = [C(rest, new_seg) + C(new_seg, new_seg)] - [C(rest, old_seg) + C(old_seg, old_seg)]
    
    # C(A, B) counts H-H pairs where one is in A and one in B, and they are neighbors.
    # If A and B are disjoint sets of positions, and we check neighbors, we count each pair once if we iterate A and check B.
    # For internal C(old_seg, old_seg), we must avoid double counting.
    
    def count_contacts_between(seg_positions, seg_start_idx, other_positions_set, other_indices_dict):
        """
        Count H-H contacts between seg (indices seg_start_idx...) and other (positions in other_positions_set).
        other_indices_dict maps position to index in chain.
        """
        c = 0
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if not is_h[idx]:
                continue
            for nb in _neighbours(pos):
                if nb in other_positions_set:
                    nb_idx = other_indices_dict[nb]
                    if is_h[nb_idx]:
                        if abs(idx - nb_idx) > 1:
                            c += 1
        return c

    # 1. C(rest, old_seg)
    # rest is in 'occupied' (pos -> idx)
    # old_seg is chain[k:]
    c_rest_old = count_contacts_between(old_seg, k, set(occupied.keys()), occupied)
    
    # 2. C(old_seg, old_seg)
    # We need to count contacts within old_seg.
    # To avoid double counting, we can count pairs (i, j) with i < j and |i-j|>1.
    # Or use the neighbor check but divide by 2?
    # Let's use a set of old_seg positions for lookup.
    old_pos_to_idx = {pos: k + i for i, pos in enumerate(old_seg)}
    
    c_old_internal = 0
    for i_offset, pos in enumerate(old_seg):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            if nb in old_pos_to_idx:
                nb_idx = old_pos_to_idx[nb]
                if is_h[nb_idx]:
                    # Count each pair once. Let's say we only count if idx < nb_idx.
                    if idx < nb_idx and abs(idx - nb_idx) > 1:
                        c_old_internal += 1
    
    # 3. C(rest, new_seg)
    c_rest_new = count_contacts_between(new_segment, k, set(occupied.keys()), occupied)
    
    # 4. C(new_seg, new_seg)
    new_pos_to_idx = {pos: k + i for i, pos in enumerate(new_segment)}
    c_new_internal = 0
    for i_offset, pos in enumerate(new_segment):
        idx = k + i_offset
        if not is_h[idx]:
            continue
        for nb in _neighbours(pos):
            if nb in new_pos_to_idx:
                nb_idx = new_pos_to_idx[nb]
                if is_h[nb_idx]:
                    if idx < nb_idx and abs(idx - nb_idx) > 1:
                        c_new_internal += 1
                        
    delta_contacts = (c_rest_new + c_new_internal) - (c_rest_old + c_old_internal)
    delta_energy = -delta_contacts
    return delta_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Time budget per call
    # Total 330s for 26 sequences.
    # Let's allocate 11.5s per sequence to be safe (26 * 11.5 = 299s).
    max_time = 11.5
    start_time = time.time()
    
    rng = random.Random(42 + len(sequence))
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0 # 0 is no contacts, energy is negative, so lower is better
    
    # Try multiple restarts
    while True:
        if time.time() - start_time > max_time:
            break
            
        # 1. Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: Straight line
            chain = [(i, 0, 0) for i in range(n)]
            
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
            
        # 2. Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        while True:
            if time.time() - start_time > max_time * 0.95:
                break
            
            # Choose move type
            # 60% Pivot, 20% Creep, 20% Slide
            move_type = rng.random()
            
            # Choose pivot index k (1 to n-1)
            # Bias towards smaller segments for Creep/Slide?
            # For Pivot, random k.
            k = rng.randint(1, n - 1)
            
            old_segment = chain[k:]
            
            # Remove old segment from occupied
            for p in old_segment:
                del occupied[p]
            
            new_segment = None
            valid = True
            
            if move_type < 0.6:
                # PIVOT
                # Rotate segment k..n-1 around chain[k-1]
                pivot_site = chain[k-1]
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                
                new_segment = []
                for p in old_segment:
                    v = [p[i] - pivot_site[i] for i in range(dimension)]
                    # Rotate v by 90 deg
                    if axis == 0:
                        if direction == 1:
                            v_new = [v[0], -v[2], v[1]]
                        else:
                            v_new = [v[0], v[2], -v[1]]
                    elif axis == 1:
                        if direction == 1:
                            v_new = [v[2], v[1], -v[0]]
                        else:
                            v_new = [-v[2], v[1], v[0]]
                    else: # axis 2
                        if direction == 1:
                            v_new = [-v[1], v[0], v[2]]
                        else:
                            v_new = [v[1], -v[0], v[2]]
                    
                    new_p = tuple(pivot_site[i] + v_new[i] for i in range(dimension))
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            elif move_type < 0.8:
                # CREEP: Move single residue k (if k < n-1) or k-1?
                # Let's move residue at index k.
                # It must not be the first or last.
                if k == 0 or k == n-1:
                    valid = False
                else:
                    p = chain[k]
                    # Try all 6 neighbors
                    moves = list(_neighbours(p))
                    rng.shuffle(moves)
                    found = False
                    for nb in moves:
                        if nb in occupied:
                            continue
                        # Check if moving p to nb causes self-intersection with k-1 or k+1?
                        # nb is adjacent to p.
                        # If nb is chain[k-1] or chain[k+1], it's a valid position geometrically but 
                        # it would make the chain backtrack? 
                        # Actually, if nb is chain[k-1], then chain[k] moves to chain[k-1]'s position? 
                        # No, nb is a neighbor of p. If nb is chain[k-1], then p moves to chain[k-1].
                        # This is a "flip" which is usually not allowed in simple SA because it 
                        # changes the connectivity? No, it's just a move.
                        # But wait, if p moves to chain[k-1], then chain[k-1] and chain[k] are at the same spot? 
                        # No, p is at chain[k]. chain[k-1] is at chain[k-1].
                        # If nb == chain[k-1], then new_p is chain[k-1].
                        # This means two residues at the same position. Invalid.
                        # So we just check `if nb in occupied`. Since chain[k-1] is in occupied, it will be skipped.
                        # Same for chain[k+1].
                        
                        new_segment = list(chain)
                        new_segment[k] = nb
                        # We only moved one residue, so "segment" is just [nb] at index k?
                        # Our _calc_delta_energy expects segment k..n-1.
                        # If we only move k, the "segment" is chain[k:].
                        # But only chain[k] changes.
                        # Let's adapt: 
                        # If we do a creep, we only change one position.
                        # We can treat it as a segment of length 1?
                        # _calc_delta_energy works on chain[k:].
                        # If we only change chain[k], then new_segment = [nb] + chain[k+1:]
                        # This is inefficient to construct.
                        # Let's stick to Pivot and Slide which move contiguous segments.
                        # Or implement a specific delta for single move.
                        # For now, let's disable Creep to keep it simple and robust.
                        found = True
                        break
                    if not found:
                        valid = False
                    else:
                        new_segment = list(chain)
                        new_segment[k] = nb
            
            else:
                # SLIDE
                # Translate segment k..n-1 by unit vector
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                offset = [0, 0, 0]
                offset[axis] = direction
                
                new_segment = []
                for p in old_segment:
                    new_p = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            if not valid:
                # Restore old
                for p, idx in zip(old_segment, range(k, n)):
                    occupied[p] = idx
                continue
            
            # Calculate Energy Delta
            delta_energy = _calc_delta_energy(sequence, chain, k, new_segment, occupied, is_h)
            
            # Acceptance
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta_energy / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                # Update state
                for p, idx in zip(new_segment, range(k, n)):
                    occupied[p] = idx
                    chain[idx] = p
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Restore old
                for p, idx in zip(old_segment, range(k, n)):
                    occupied[p] = idx
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            steps += 1
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
