import time
import random

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate new H-H contacts when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    x, y, z = q
    delta = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _compute_free_space(occupied, q):
    """Count free neighboring positions."""
    x, y, z = q
    free = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r not in occupied:
            free += 1
    return free

def _compute_full_energy(sequence, chain):
    """Compute total H-H contact energy (negative number)."""
    n = len(chain)
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    energy = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            r = (x + dx, y + dy, z + dz)
            if r in occupied:
                j = occupied[r]
                if j > i and j != i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def _beam_search_fold(sequence, n, deadline, beam_width=50):
    """
    Beam search: grow chain step by step, keeping only the best beam_width partial paths.
    Score = contacts * 10 + free_space * 1 - distance * 0.05
    """
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0.0)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, score in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                new_score = score
                if sequence[i] == 'H':
                    new_score += _calc_delta_energy(sequence, occupied, i, q) * 10.0
                new_score += _compute_free_space(occupied, q) * 1.0
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                new_score -= dist * 0.05
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, score in beams:
        if len(chain) == n:
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization with incremental energy updates and segment shift moves.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_full_energy(sequence, chain)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    # Precompute H positions for faster contact counting
    h_positions = set(i for i in range(n) if sequence[i] == 'H')
    
    def _get_energy():
        energy = 0
        for i in range(n):
            if i not in h_positions:
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1 and j in h_positions:
                        energy -= 1
        return energy
    
    def _delta_energy_for_move(i, old_pos, new_pos):
        """Compute energy change when moving residue i from old_pos to new_pos."""
        delta = 0
        # Remove old contacts involving residue i
        if i in h_positions:
            x, y, z = old_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                        delta += 1  # removing a contact
        # Add new contacts involving residue i
        if i in h_positions:
            x, y, z = new_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                        delta -= 1  # adding a contact
        return delta
    
    current_energy = _get_energy()
    improved = True
    stall_count = 0
    max_stall = 200
    
    while improved and time.time() < deadline:
        improved = False
        stall_count = 0
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            if n < 4:
                break
            
            # Randomly choose between single-residue move and segment shift
            if rng.random() < 0.6:
                # Single residue move
                i = rng.randint(1, n - 2)
                old_pos = chain[i]
                valid_mvs = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                    if q not in occupied or occupied[q] == i:
                        # Check connectivity to neighbors
                        if i > 0:
                            prev = chain[i-1]
                            if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                                continue
                        if i < n-1:
                            nxt = chain[i+1]
                            if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                                continue
                        if q not in occupied or occupied[q] == i:
                            valid_mvs.append(q)
                
                if not valid_mvs:
                    stall_count += 1
                    continue
                
                new_pos = rng.choice(valid_mvs)
                if new_pos == old_pos:
                    continue
                
                delta_e = _delta_energy_for_move(i, old_pos, new_pos)
                
                if delta_e <= 0:
                    # Accept move (improves or equal energy)
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    current_energy += delta_e
                    improved = True
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > max_stall:
                        break
            else:
                # Segment shift: move a small segment by one step
                seg_len = rng.randint(2, min(6, n // 4)) if n >= 8 else 2
                start = rng.randint(1, n - seg_len - 1)
                end = start + seg_len - 1
                
                # Try shifting the segment in one of 6 directions
                dir_choice = rng.choice(_NEIGHBOR_OFFSETS)
                dx, dy, dz = dir_choice
                
                # Check if all positions in shifted segment are valid
                valid = True
                new_positions = []
                for idx in range(start, end + 1):
                    new_p = (chain[idx][0] + dx, chain[idx][1] + dy, chain[idx][2] + dz)
                    if new_p in occupied and occupied[new_p] not in range(start, end + 1):
                        valid = False
                        break
                    new_positions.append(new_p)
                
                if not valid:
                    stall_count += 1
                    continue
                
                # Check connectivity at boundaries
                if start > 0:
                    prev = chain[start - 1]
                    first_new = new_positions[0]
                    if abs(first_new[0]-prev[0]) + abs(first_new[1]-prev[1]) + abs(first_new[2]-prev[2]) != 1:
                        stall_count += 1
                        continue
                if end < n - 1:
                    nxt = chain[end + 1]
                    last_new = new_positions[-1]
                    if abs(last_new[0]-nxt[0]) + abs(last_new[1]-nxt[1]) + abs(last_new[2]-nxt[2]) != 1:
                        stall_count += 1
                        continue
                
                # Compute energy change
                delta_e = 0
                # Remove old contacts for residues in segment
                for idx in range(start, end + 1):
                    if idx in h_positions:
                        x, y, z = chain[idx]
                        for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                            r = (x + ddx, y + ddy, z + ddz)
                            if r in occupied:
                                j = occupied[r]
                                if j not in range(start, end + 1) and j != idx - 1 and j != idx + 1 and j in h_positions:
                                    delta_e += 1
                
                # Add new contacts for residues in segment
                for idx in range(start, end + 1):
                    if idx in h_positions:
                        x, y, z = new_positions[idx - start]
                        for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                            r = (x + ddx, y + ddy, z + ddz)
                            if r in occupied:
                                j = occupied[r]
                                if j not in range(start, end + 1) and j != idx - 1 and j != idx + 1 and j in h_positions:
                                    delta_e -= 1
                
                if delta_e <= 0:
                    # Accept segment shift
                    for idx in range(start, end + 1):
                        del occupied[chain[idx]]
                    for idx, new_p in zip(range(start, end + 1), new_positions):
                        occupied[new_p] = idx
                        chain[idx] = new_p
                    current_energy += delta_e
                    improved = True
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > max_stall:
                        break
    
    return chain, current_energy

def _greedy_complete(sequence, chain, occupied, n, deadline):
    """Greedily complete a partial chain. Returns None if stuck."""
    for j in range(len(chain), n):
        if time.time() > deadline:
            return None
        last_pos = chain[-1]
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in occupied:
                valid_mvs.append(q)
        if not valid_mvs:
            return None
        # Pick move maximizing H-contacts, then free space
        best_q = None
        best_score = -1
        for q in valid_mvs:
            score = _calc_delta_energy(sequence, occupied, j, q)
            free = 0
            x, y, z = q
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r not in occupied:
                    free += 1
            score += free * 0.1
            if score > best_score:
                best_score = score
                best_q = q
        if best_q is None:
            return None
        chain.append(best_q)
        occupied[best_q] = j
    return chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 220.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    # Adaptive beam configs: start with moderate width, adjust based on time
    beam_configs = [
        (60, 42), (100, 123), (140, 456), (80, 789), (120, 321),
        (90, 654), (110, 987), (70, 111), (130, 222), (85, 333)
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        energy = _compute_full_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        # Local optimization with multiple restarts
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        # Additional local optimization passes on the best solution
        for _ in range(2):
            if time.time() > deadline:
                break
            rng_local2 = random.Random(rng.randint(0, 100000))
            opt_chain2, opt_energy2 = _local_optimize(sequence, best_chain[:], deadline, rng_local2)
            if opt_energy2 < best_energy:
                best_energy = opt_energy2
                best_chain = opt_chain2[:]
        
        # Early termination if we're doing well
        if n > 10 and best_energy <= -n * 0.5:
            break
    
    # Final polish: more aggressive local optimization on the best chain
    if best_chain is not None:
        for round_num in range(8):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num * 7)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        # Fallback: try greedy completion from start
        init_chain = [(0, 0, 0)]
        init_occupied = {(0, 0, 0): 0}
        completed = _greedy_complete(sequence, init_chain, init_occupied, n, deadline)
        if completed is not None and len(completed) == n:
            best_chain = completed
        else:
            best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
