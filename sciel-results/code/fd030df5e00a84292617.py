import time
import random
import math
import heapq

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate H-H contact energy change when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    x, y, z = q
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

def _compute_energy(sequence, chain):
    """Compute total H-H contact energy (negative) for a full chain."""
    n = len(sequence)
    if n < 2:
        return 0
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

def _max_future_contacts(sequence, i, n):
    """
    Admissible heuristic: maximum possible H-H contacts from position i onward.
    Each H residue can form at most 6 contacts, but realistically limited by
    remaining H count and spatial constraints.
    """
    remaining_h = sum(1 for k in range(i, n) if sequence[k] == 'H')
    # Each H can contact at most (remaining_h - 1) other H's, but spatially limited
    # Conservative: each H gets at most 4 useful contacts on average in dense packing
    return min(remaining_h * (remaining_h - 1) // 2, remaining_h * 4)

def _beam_search_fold(sequence, n, deadline, rng, beam_width=50):
    """
    A*-style beam search with admissible heuristic for future contacts.
    Priority = current_contacts * 10 + free_space * 1 - dist * 0.05 + heuristic_bonus
    """
    # Start with residue 0 at origin
    start_occupied = {(0, 0, 0): 0}
    start_chain = [(0, 0, 0)]
    
    # Priority queue: (-priority, counter, chain, occupied, contacts)
    counter = 0
    h_count = sum(1 for c in sequence if c == 'H')
    initial_heuristic = _max_future_contacts(sequence, 1, n)
    initial_priority = 0.0 + 6.0 * 1.0 - 0.0 * 0.05 + initial_heuristic * 0.5
    heapq.heappush(heapq.heapq if False else [], 0)  # placeholder
    
    # Use a simple list-based beam for clarity and speed
    beams = [([0, 0, 0] if False else [(0, 0, 0)], {(0, 0, 0): 0}, 0.0, initial_heuristic)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, contacts, remaining_heuristic in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                new_contacts = contacts
                if sequence[i] == 'H':
                    new_contacts += _calc_delta_energy(sequence, occupied, i, q)
                
                free = _compute_free_space(occupied, q)
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                
                # Admissible heuristic for remaining sequence
                rem_h = sum(1 for k in range(i + 1, n) if sequence[k] == 'H')
                # Each remaining H can form at most min(rem_h, 4) useful contacts
                # This is admissible (never overestimates)
                future_heuristic = min(rem_h * (rem_h - 1) // 2, rem_h * 4)
                
                # Priority: favor more current contacts, more free space, compactness
                # and high potential for future contacts
                priority = new_contacts * 10.0 + free * 1.0 - dist * 0.05 + future_heuristic * 0.3
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_contacts, future_heuristic, priority))
        
        if not new_beams:
            break
        
        # Keep only top beam_width by priority
        new_beams.sort(key=lambda x: x[4], reverse=True)
        beams = [(b[0], b[1], b[2], b[3]) for b in new_beams[:beam_width]]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, contacts, _ in beams:
        if len(chain) == n:
            score = contacts * 10.0
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization with simulated annealing and incremental energy updates.
    Uses single-residue moves and 2-residue moves.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    is_h = [1 if c == 'H' else 0 for c in sequence]
    
    def _compute_full_energy():
        energy = 0
        for i in range(n):
            if not is_h[i]:
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1 and is_h[j]:
                        energy -= 1
        return energy
    
    current_energy = _compute_full_energy()
    best_energy = current_energy
    best_chain = chain[:]
    
    def _delta_energy_single_move(i, old_pos, new_pos):
        """
        Compute energy delta for moving residue i from old_pos to new_pos.
        Positive delta means energy increased (worse), negative means decreased (better).
        """
        delta = 0
        if is_h[i]:
            # Remove contacts at old_pos
            ox, oy, oz = old_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (ox + dx, oy + dy, oz + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and is_h[j]:
                        delta += 1
            # Add contacts at new_pos
            nx, ny, nz = new_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (nx + dx, ny + dy, nz + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and is_h[j]:
                        delta -= 1
        return delta
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.01
    total_iterations = 5000
    iterations_per_pass = 100
    
    current_temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / total_iterations)
    
    improved = True
    stall_count = 0
    iter_count = 0
    
    while iter_count < total_iterations and time.time() < deadline:
        iter_count += 1
        current_temp *= cooling_rate
        
        # Try single-residue moves
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
            if q in occupied and occupied[q] != i:
                continue
            # Check connectivity with neighbors
            if i > 0:
                prev = chain[i-1]
                if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                    continue
            if i < n-1:
                nxt = chain[i+1]
                if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                    continue
            valid_mvs.append(q)
        
        if not valid_mvs:
            continue
        
        new_pos = rng.choice(valid_mvs)
        if new_pos == old_pos:
            continue
        
        d = _delta_energy_single_move(i, old_pos, new_pos)
        
        # Metropolis acceptance
        accept = False
        if d < 0:
            accept = True
        elif current_temp > 0:
            if rng.random() < math.exp(-d / current_temp):
                accept = True
        
        if accept:
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            current_energy += d
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        # Periodically try 2-residue moves
        if n >= 6 and iter_count % 5 == 0 and time.time() < deadline:
            i2 = rng.randint(1, n - 3)
            j_candidates = [k for k in range(1, n-1) if abs(k - i2) > 1]
            if j_candidates:
                j2 = rng.choice(j_candidates)
                
                old_i = chain[i2]
                old_j = chain[j2]
                
                valid_i = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (old_i[0] + dx, old_i[1] + dy, old_i[2] + dz)
                    if q in occupied and occupied[q] != i2:
                        continue
                    if i2 > 0:
                        prev = chain[i2-1]
                        if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                            continue
                    if i2 < n-1:
                        nxt = chain[i2+1]
                        if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                            continue
                    valid_i.append(q)
                
                if valid_i:
                    new_i = rng.choice(valid_i)
                    if new_i != old_i:
                        valid_j = []
                        for dx, dy, dz in _NEIGHBOR_OFFSETS:
                            q = (old_j[0] + dx, old_j[1] + dy, old_j[2] + dz)
                            if q in occupied:
                                occ_idx = occupied[q]
                                if occ_idx == i2:
                                    if q == old_i:
                                        pass
                                    elif q == new_i:
                                        continue
                                    else:
                                        continue
                                elif occ_idx == j2:
                                    pass
                                else:
                                    continue
                            if j2 > 0:
                                prev = chain[j2-1]
                                if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                                    continue
                            if j2 < n-1:
                                nxt = chain[j2+1]
                                if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                                    continue
                            valid_j.append(q)
                        
                        if valid_j:
                            new_j = rng.choice(valid_j)
                            if new_j != old_j and new_i != new_j:
                                if new_i not in occupied or occupied[new_i] == j2:
                                    if new_j not in occupied or occupied[new_j] == i2:
                                        # Compute combined delta
                                        delta2 = 0
                                        if is_h[i2]:
                                            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                                                r = (old_i[0]+dx, old_i[1]+dy, old_i[2]+dz)
                                                if r in occupied:
                                                    k = occupied[r]
                                                    if k != i2 and k != j2 and k != i2-1 and k != i2+1 and is_h[k]:
                                                        delta2 += 1
                                            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                                                r = (new_i[0]+dx, new_i[1]+dy, new_i[2]+dz)
                                                if r in occupied:
                                                    k = occupied[r]
                                                    if k != i2 and k != j2 and k != i2-1 and k != i2+1 and is_h[k]:
                                                        delta2 -= 1
                                                elif r == new_j and is_h[j2] and abs(i2-j2) != 1:
                                                    delta2 -= 1
                                        if is_h[j2]:
                                            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                                                r = (old_j[0]+dx, old_j[1]+dy, old_j[2]+dz)
                                                if r in occupied:
                                                    k = occupied[r]
                                                    if k != j2 and k != i2 and k != j2-1 and k != j2+1 and is_h[k]:
                                                        delta2 += 1
                                            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                                                r = (new_j[0]+dx, new_j[1]+dy, new_j[2]+dz)
                                                if r in occupied:
                                                    k = occupied[r]
                                                    if k != j2 and k != i2 and k != j2-1 and k != j2+1 and is_h[k]:
                                                        delta2 -= 1
                                                elif r == new_i and is_h[i2] and abs(i2-j2) != 1:
                                                    delta2 -= 1
                                        
                                        accept2 = False
                                        if delta2 < 0:
                                            accept2 = True
                                        elif current_temp > 0:
                                            if rng.random() < math.exp(-delta2 / current_temp):
                                                accept2 = True
                                        
                                        if accept2:
                                            del occupied[old_i]
                                            del occupied[old_j]
                                            occupied[new_i] = i2
                                            occupied[new_j] = j2
                                            chain[i2] = new_i
                                            chain[j2] = new_j
                                            current_energy += delta2
                                            
                                            if current_energy < best_energy:
                                                best_energy = current_energy
                                                best_chain = chain[:]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple beam search runs with different parameters
    beam_configs = [
        (60, 42), (80, 123), (100, 456), (70, 789), (90, 321),
        (50, 654), (110, 987), (65, 111), (75, 222), (85, 333)
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, rng_local, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        # Local optimization with simulated annealing
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        # Early termination if we're doing well
        if n > 10 and best_energy <= -n * 0.5:
            break
    
    # Additional polish passes on the best solution
    if best_chain is not None:
        for round_num in range(5):
            if time.time() > deadline:
                break
            rng_local = random.Random(5000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
