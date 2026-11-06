import time
import random
import math

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

def _polish(sequence, chain, deadline):
    """
    Hill climbing polish: repeatedly move residues to improve energy until no improvement is found.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_full_energy(sequence, chain)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    h_positions = set(i for i in range(n) if sequence[i] == 'H')
    
    current_energy = _compute_full_energy(sequence, chain)
    improved = True
    steps = 0
    max_steps = n * 20 # Safety cap
    
    while improved and time.time() < deadline and steps < max_steps:
        improved = False
        steps += 1
        for i in range(1, n - 1):
            if time.time() > deadline:
                return chain, current_energy
            old_pos = chain[i]
            best_new_pos = None
            best_delta = 0
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if q == old_pos:
                    continue
                if q in occupied and occupied[q] != i:
                    continue
                # Check connectivity
                if i > 0:
                    prev = chain[i-1]
                    if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                        continue
                if i < n-1:
                    nxt = chain[i+1]
                    if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                        continue
                
                # Calculate delta energy
                delta = 0
                if i in h_positions:
                    # Remove old contacts
                    x, y, z = old_pos
                    for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                        r = (x + ddx, y + ddy, z + ddz)
                        if r in occupied:
                            j = occupied[r]
                            if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                                delta += 1
                    # Add new contacts
                    x, y, z = q
                    for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                        r = (x + ddx, y + ddy, z + ddz)
                        if r in occupied:
                            j = occupied[r]
                            if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                                delta -= 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_pos = q
            
            if best_new_pos is not None and best_delta < 0:
                # Apply move
                del occupied[old_pos]
                occupied[best_new_pos] = i
                chain[i] = best_new_pos
                current_energy += best_delta
                improved = True
                
    return chain, current_energy

def _local_optimize_sa(sequence, chain, deadline, rng, t_start=10.0, t_end=0.1, cooling_rate=0.995):
    """
    Local optimization using Simulated Annealing.
    Moves: Single residue, Segment Shift, Segment Reversal.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_full_energy(sequence, chain)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
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
        if i in h_positions:
            x, y, z = old_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                        delta += 1
        if i in h_positions:
            x, y, z = new_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and j in h_positions:
                        delta -= 1
        return delta

    def _delta_energy_segment_shift(start, end, dx, dy, dz):
        """Compute energy change for shifting segment [start, end] by (dx, dy, dz)."""
        delta = 0
        # Remove old contacts for residues in segment
        for idx in range(start, end + 1):
            if idx in h_positions:
                x, y, z = chain[idx]
                for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                    r = (x + ddx, y + ddy, z + ddz)
                    if r in occupied:
                        j = occupied[r]
                        if j not in range(start, end + 1) and j != idx - 1 and j != idx + 1 and j in h_positions:
                            delta += 1
        
        # Add new contacts for residues in segment
        # Note: We must check against the "virtual" new positions
        # But we haven't updated occupied yet.
        # So we check neighbors in the NEW position against:
        # 1. Other residues in the segment (in their new positions)
        # 2. Residues outside the segment (in their old positions)
        
        new_pos_map = {}
        for idx in range(start, end + 1):
            new_pos_map[idx] = (chain[idx][0] + dx, chain[idx][1] + dy, chain[idx][2] + dz)
            
        for idx in range(start, end + 1):
            if idx in h_positions:
                x, y, z = new_pos_map[idx]
                for ddx, ddy, ddz in _NEIGHBOR_OFFSETS:
                    r = (x + ddx, y + ddy, z + ddz)
                    # Check if r is occupied by a residue outside the segment
                    if r in occupied:
                        j = occupied[r]
                        if j not in range(start, end + 1) and j != idx - 1 and j != idx + 1 and j in h_positions:
                            delta -= 1
                    # Check if r is the new position of a residue inside the segment
                    # We need to check if any idx2 in [start, end] has new_pos_map[idx2] == r
                    # This is O(L^2) if done naively. For small L it's fine.
                    for idx2 in range(start, end + 1):
                        if idx2 != idx and new_pos_map[idx2] == r:
                            if idx2 != idx - 1 and idx2 != idx + 1 and idx2 in h_positions:
                                delta -= 1
        return delta

    def _delta_energy_reversal(start, end):
        """Compute energy change for reversing segment [start, end]."""
        # This is complex to calculate incrementally without applying the move.
        # For now, we might skip this or approximate.
        # Given time constraints, let's stick to shifts and single moves for SA.
        # Or, we can just apply the move, calculate full energy, and revert if rejected.
        # Full energy calculation is O(N * 6). For N=100, this is 600 ops.
        # SA step is 1 move. If we do 1000 steps, that's 600k ops. Very fast.
        # So we can just use full energy calculation for reversals.
        return None # Placeholder

    current_energy = _get_energy()
    T = t_start
    steps = 0
    max_steps = 5000 # Cap steps to avoid infinite loops if T doesn't drop fast enough
    
    # Adaptive cooling
    # We want T to reach t_end in roughly 2000-3000 steps
    # T = T_start * cooling_rate^steps
    # t_end = t_start * cooling_rate^steps
    # steps = log(t_end/t_start) / log(cooling_rate)
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        T *= cooling_rate
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if q not in occupied or occupied[q] == i:
                    # Check connectivity
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
                continue
                
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos:
                continue
                
            delta_e = _delta_energy_for_move(i, old_pos, new_pos)
            
            # Metropolis acceptance
            if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                del occupied[old_pos]
                occupied[new_pos] = i
                chain[i] = new_pos
                current_energy += delta_e
            else:
                pass # Reject, no state change

        elif move_type < 0.9:
            # Segment shift
            seg_len = rng.randint(2, min(8, n // 4)) if n >= 8 else 2
            start = rng.randint(1, n - seg_len - 1)
            end = start + seg_len - 1
            
            dir_choice = rng.choice(_NEIGHBOR_OFFSETS)
            dx, dy, dz = dir_choice
            
            # Check validity
            valid = True
            new_positions = []
            for idx in range(start, end + 1):
                new_p = (chain[idx][0] + dx, chain[idx][1] + dy, chain[idx][2] + dz)
                if new_p in occupied and occupied[new_p] not in range(start, end + 1):
                    valid = False
                    break
                new_positions.append(new_p)
            
            if not valid:
                continue
                
            # Check connectivity at boundaries
            if start > 0:
                prev = chain[start - 1]
                first_new = new_positions[0]
                if abs(first_new[0]-prev[0]) + abs(first_new[1]-prev[1]) + abs(first_new[2]-prev[2]) != 1:
                    continue
            if end < n - 1:
                nxt = chain[end + 1]
                last_new = new_positions[-1]
                if abs(last_new[0]-nxt[0]) + abs(last_new[1]-nxt[1]) + abs(last_new[2]-nxt[2]) != 1:
                    continue
            
            delta_e = _delta_energy_segment_shift(start, end, dx, dy, dz)
            
            if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                for idx in range(start, end + 1):
                    del occupied[chain[idx]]
                for idx, new_p in zip(range(start, end + 1), new_positions):
                    occupied[new_p] = idx
                    chain[idx] = new_p
                current_energy += delta_e
            else:
                pass

        else:
            # Segment Reversal (Full energy check for simplicity)
            seg_len = rng.randint(3, min(10, n // 3)) if n >= 10 else 3
            start = rng.randint(1, n - seg_len - 1)
            end = start + seg_len - 1
            
            # Apply reversal
            old_segment = chain[start:end+1]
            chain[start:end+1] = old_segment[::-1]
            
            # Check self-avoidance
            valid = True
            # Rebuild occupied for the segment
            for idx in range(start, end + 1):
                del occupied[chain[idx]]
            for idx in range(start, end + 1):
                if chain[idx] in occupied and occupied[chain[idx]] != idx:
                    valid = False
                    break
                occupied[chain[idx]] = idx
            
            if valid:
                # Check connectivity at boundaries
                if start > 0:
                    prev = chain[start - 1]
                    first_new = chain[start]
                    if abs(first_new[0]-prev[0]) + abs(first_new[1]-prev[1]) + abs(first_new[2]-prev[2]) != 1:
                        valid = False
                if end < n - 1:
                    nxt = chain[end + 1]
                    last_new = chain[end]
                    if abs(last_new[0]-nxt[0]) + abs(last_new[1]-nxt[1]) + abs(last_new[2]-nxt[2]) != 1:
                        valid = False
                
                if valid:
                    new_energy = _get_energy()
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                        current_energy = new_energy
                    else:
                        # Revert
                        chain[start:end+1] = old_segment
                        for idx in range(start, end + 1):
                            del occupied[chain[idx]]
                        for idx in range(start, end + 1):
                            occupied[chain[idx]] = idx
                else:
                    # Revert
                    chain[start:end+1] = old_segment
                    for idx in range(start, end + 1):
                        del occupied[chain[idx]]
                    for idx in range(start, end + 1):
                        occupied[chain[idx]] = idx
            else:
                # Revert
                chain[start:end+1] = old_segment
                for idx in range(start, end + 1):
                    del occupied[chain[idx]]
                for idx in range(start, end + 1):
                    occupied[chain[idx]] = idx

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
    deadline = start_time + 300.0 # 330s limit, keep 30s margin
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    # Adaptive beam configs
    beam_configs = [
        (50, 42), (80, 123), (100, 456), (60, 789), (90, 321),
        (70, 654), (85, 987), (55, 111), (95, 222), (65, 333)
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
        
        # Local optimization with SA
        opt_chain, opt_energy = _local_optimize_sa(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        # Additional SA passes on the best solution
        for _ in range(3):
            if time.time() > deadline:
                break
            rng_local2 = random.Random(rng.randint(0, 100000))
            opt_chain2, opt_energy2 = _local_optimize_sa(sequence, best_chain[:], deadline, rng_local2)
            if opt_energy2 < best_energy:
                best_energy = opt_energy2
                best_chain = opt_chain2[:]
        
        # Early termination if we're doing well
        if n > 10 and best_energy <= -n * 0.6:
            break
    
    # Final polish: more aggressive SA on the best chain
    if best_chain is not None:
        for round_num in range(10):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num * 7)
            opt_chain, opt_energy = _local_optimize_sa(sequence, best_chain[:], deadline, rng_local, t_start=5.0, t_end=0.01, cooling_rate=0.99)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
        
        # Final polish
        if time.time() < deadline:
            polished_chain, polished_energy = _polish(sequence, best_chain[:], deadline)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished_chain

    if best_chain is None:
        # Fallback
        init_chain = [(0, 0, 0)]
        init_occupied = {(0, 0, 0): 0}
        completed = _greedy_complete(sequence, init_chain, init_occupied, n, deadline)
        if completed is not None and len(completed) == n:
            best_chain = completed
        else:
            best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
