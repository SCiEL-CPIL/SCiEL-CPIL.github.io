import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _beam_search(sequence, dimension, beam_width=400):
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    start_occ = {start}
    start_pos_idx = {start: 0}
    start_energy = 0
    start_free = 6
    
    beam = [(0, [start], start_occ, start_pos_idx, start_energy, start_free)]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        candidates = []
        last_residue_is_h = (sequence[i] == 'H')
        
        for score, chain, occ, pos_idx, energy, free in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                # Calculate delta energy
                delta_e = 0
                if last_residue_is_h:
                    for odx, ody, odz in _OFFSETS:
                        neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                        if neighbor in occ:
                            j = pos_idx[neighbor]
                            if j < i - 1:
                                if sequence[j] == 'H':
                                    delta_e -= 1
                            
                new_energy = energy + delta_e
                
                # Calculate free space for new_point
                new_free = 0
                for odx, ody, odz in _OFFSETS:
                    neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                    if neighbor not in occ:
                        new_free += 1
                
                # Compactness: prefer moves that reduce bounding box volume
                # Use Manhattan distance from origin as proxy for compactness
                compactness = abs(new_point[0]) + abs(new_point[1]) + abs(new_point[2])
                
                # Score: energy primary, compactness secondary, free space tertiary
                # Lower score is better
                noise = random.random() * 0.005
                new_score = new_energy - 0.05 * new_free + 0.02 * compactness + noise
                
                # Prepare new state
                new_chain = chain + [new_point]
                new_occ = set(occ)
                new_occ.add(new_point)
                new_pos_idx = dict(pos_idx)
                new_pos_idx[new_point] = i
                
                candidates.append((new_score, new_chain, new_occ, new_pos_idx, new_energy, new_free))
        
        if not candidates:
            return None
            
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = candidates[:beam_width]
        
    if not beam:
        return None
        
    # Return the best chain by actual energy, not score
    best = min(beam, key=lambda x: x[4])
    return best[1]

def _calc_energy(chain, sequence):
    """Calculate the total energy of a chain configuration."""
    n = len(chain)
    if n < 2:
        return 0
    
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    e = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _OFFSETS:
            nb = (x+dx, y+dy, z+dz)
            if nb in occ:
                j = pos_to_idx[nb]
                if abs(i - j) > 1:
                    if sequence[j] == 'H':
                        e -= 1
    return e // 2

def _local_search(chain, sequence, dimension, time_budget=None):
    """
    Perform local search on the chain to improve energy using pivot moves.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    chain = [tuple(p) for p in chain]
    
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Precompute H indices
    h_indices = set(i for i in range(n) if sequence[i] == 'H')
    
    # Rotation functions
    def rotate_point(p, axis, angle):
        x, y, z = p
        if axis == 0:  # x-axis
            if angle == 1:
                return (x, -z, y)
            elif angle == 2:
                return (x, -y, -z)
            else:
                return (x, z, -y)
        elif axis == 1:  # y-axis
            if angle == 1:
                return (z, y, -x)
            elif angle == 2:
                return (-x, y, -z)
            else:
                return (-z, y, x)
        else:  # z-axis
            if angle == 1:
                return (y, -x, z)
            elif angle == 2:
                return (-x, -y, z)
            else:
                return (-y, x, z)
    
    it = 0
    max_iters = 50000
    no_improve_count = 0
    patience = 2000
    
    while it < max_iters:
        if _is_deadline_exceeded():
            break
        if time_budget is not None and time.time() > time_budget:
            break
        
        it += 1
        
        # Pick a random pivot point (not at ends)
        pivot = random.randint(1, n - 2)
        
        # Pick rotation axis and angle
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])
        
        # Extract subchain
        subchain = chain[pivot+1:]
        if not subchain:
            continue
        
        # Apply rotation to subchain
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]
        
        # Check pivot connection validity
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            continue
        
        # Check self-avoidance: new positions must not conflict with existing (non-subchain) positions
        old_positions = set(subchain)
        new_positions = set(new_subchain)
        
        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Check that the last of new_subchain doesn't conflict (already covered above)
        
        # Apply move
        for p in old_positions:
            occ.remove(p)
            del pos_to_idx[p]
        
        for i, p in enumerate(new_subchain):
            occ.add(p)
            pos_to_idx[p] = pivot + 1 + i
        
        chain[pivot+1:] = new_subchain
        
        # Calculate new energy
        new_energy = _calc_energy(chain, sequence)
        
        if new_energy < current_energy:
            current_energy = new_energy
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
        else:
            no_improve_count += 1
            # Revert the move
            for p in new_positions:
                occ.remove(p)
                del pos_to_idx[p]
            
            for i, p in enumerate(old_positions):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i
            
            chain[pivot+1:] = subchain
        
        # Early stopping if no improvement for a while
        if no_improve_count > patience:
            break
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    # Allocate time: 55% beam search, 40% local search, 5% buffer
    total_budget = 330 * _BUDGET_FRACTION
    beam_time = total_budget * 0.55
    local_search_time = total_budget * 0.40
    
    beam_deadline = time.time() + beam_time
    local_search_deadline = time.time() + beam_time + local_search_time
    
    # Override deadline for beam search phase
    global _DEADLINE
    old_deadline = _DEADLINE
    _DEADLINE = beam_deadline
    
    chain = _beam_search(sequence, dimension, beam_width=400)
    
    # Restore deadline
    _DEADLINE = old_deadline
    
    if chain is None:
        # Fallback: simple greedy walk
        chain = [(0, 0, 0)]
        occ = {chain[0]}
        for i in range(1, n):
            last = chain[-1]
            found = False
            for dx, dy, dz in _OFFSETS:
                np = (last[0]+dx, last[1]+dy, last[2]+dz)
                if np not in occ:
                    chain.append(np)
                    occ.add(np)
                    found = True
                    break
            if not found:
                break
        if len(chain) < n:
            # Trapped, return what we have (shouldn't happen often)
            return [list(p) for p in chain]
    
    # Local search phase
    _DEADLINE = local_search_deadline
    refined_chain = _local_search(chain, sequence, dimension, time_budget=local_search_deadline)
    
    # Restore deadline
    _DEADLINE = old_deadline
    
    return [list(p) for p in refined_chain]
