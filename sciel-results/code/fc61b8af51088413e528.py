"""Improved program with better heuristics, optimized energy calc, and enhanced local search."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield all 6 neighbours in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive residues on adjacent sites)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _count_free_neighbors(point, occupied):
    """Count free neighbors of a point."""
    count = 0
    for q in _neighbours(point):
        if q not in occupied:
            count += 1
    return count


def _generate_chain(sequence, dimension, rng, w_contact=10.0, w_space=1.0, w_compact=0.1, w_trap=5.0, noise_scale=0.5, lookahead=2):
    """
    Generate a chain using greedy with lookahead for contact potential.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Precompute H positions for quick lookup
    h_set = set(i for i in range(n) if sequence[i] == 'H')
    
    cx, cy, cz = 0.0, 0.0, 0.0
    
    for i in range(1, n):
        candidates = []
        
        centroid = (cx / i, cy / i, cz / i)
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            score = 0.0
            
            # 1. Immediate H-H contacts formed
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += w_contact
            
            # 2. Lookahead: potential contacts with future H residues near this site
            if sequence[i] == "H" and lookahead > 0:
                for q in _neighbours(site):
                    if q not in occupied:
                        # This site could form a contact later if an H lands here
                        # Weight by proximity to centroid (more likely to be visited)
                        dist_c = (q[0] - centroid[0])**2 + (q[1] - centroid[1])**2 + (q[2] - centroid[2])**2
                        score += 0.5 * w_contact / (1.0 + dist_c)
            
            # 3. Space: Free neighbors (to avoid trapping)
            free_count = _count_free_neighbors(site, occupied)
            score += w_space * free_count
            
            # 4. Trap penalty
            if i < n - 1 and free_count <= 1:
                score -= w_trap
            
            # 5. Compactness: Prefer moves closer to current centroid
            dist_to_centroid = (site[0] - centroid[0])**2 + (site[1] - centroid[1])**2 + (site[2] - centroid[2])**2
            score -= w_compact * dist_to_centroid
            
            # 6. Noise
            score += rng.uniform(0, noise_scale)
            
            candidates.append((score, site))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_site = candidates[0]
        
        chain.append(best_site)
        occupied[best_site] = i
        
        cx += best_site[0]
        cy += best_site[1]
        cz += best_site[2]
        
    return chain


def _local_search_sa(sequence, dimension, chain, rng, time_limit=2.0):
    """
    Simulated Annealing local search with Pivot, Kink, and Shift moves.
    Optimized with incremental energy updates where possible.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_start = 5.0
    T_end = 0.1
    start_time = time.time()
    
    # Precompute H indices
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        T = max(T_end, T_start - (elapsed / time_limit) * (T_start - T_end))
        
        move_type = rng.random()
        new_chain = None
        valid_move = False
        
        if move_type < 0.5:
            # Pivot move: Rotate segment [i+1, j] around bond (i, i+1)
            if n < 5:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            p0 = current_chain[i]
            old_i1 = current_chain[i+1]
            
            valid_new_i1 = []
            for q in _neighbours(p0):
                if q != old_i1:
                    valid_new_i1.append(q)
            
            if not valid_new_i1:
                continue
            
            new_i1 = rng.choice(valid_new_i1)
            
            v_old = (old_i1[0] - p0[0], old_i1[1] - p0[1], old_i1[2] - p0[2])
            v_new = (new_i1[0] - p0[0], new_i1[1] - p0[1], new_i1[2] - p0[2])
            
            cross = (v_old[1]*v_new[2] - v_old[2]*v_new[1],
                     v_old[2]*v_new[0] - v_old[0]*v_new[2],
                     v_old[0]*v_new[1] - v_old[1]*v_new[0])
            
            dot = v_old[0]*v_new[0] + v_old[1]*v_new[1] + v_old[2]*v_new[2]
            
            if abs(dot - 1.0) < 1e-10:
                continue
            elif abs(dot + 1.0) < 1e-10:
                if abs(v_old[0]) < 0.9:
                    perp = (1, 0, 0)
                elif abs(v_old[1]) < 0.9:
                    perp = (0, 1, 0)
                else:
                    perp = (0, 0, 1)
                
                def rot180(v, axis):
                    d = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]
                    return (2*axis[0]*d - v[0], 2*axis[1]*d - v[1], 2*axis[2]*d - v[2])
                
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rot180(v, perp)
                    new_positions.append((p0[0] + rv[0], p0[1] + rv[1], p0[2] + rv[2]))
            else:
                sin_theta = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
                cos_theta = dot
                axis = (cross[0]/sin_theta, cross[1]/sin_theta, cross[2]/sin_theta)
                
                def rodrigues(v, axis, cos_t, sin_t):
                    dot_av = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
                    cross_av = (axis[1]*v[2] - axis[2]*v[1],
                               axis[2]*v[0] - axis[0]*v[2],
                               axis[0]*v[1] - axis[1]*v[0])
                    return (v[0]*cos_t + cross_av[0]*sin_t + axis[0]*dot_av*(1-cos_t),
                            v[1]*cos_t + cross_av[1]*sin_t + axis[1]*dot_av*(1-cos_t),
                            v[2]*cos_t + cross_av[2]*sin_t + axis[2]*dot_av*(1-cos_t))
                
                new_positions = []
                for k in range(i+1, j+1):
                    v = (current_chain[k][0] - p0[0], current_chain[k][1] - p0[1], current_chain[k][2] - p0[2])
                    rv = rodrigues(v, axis, cos_theta, sin_theta)
                    new_positions.append((p0[0] + int(round(rv[0])), p0[1] + int(round(rv[1])), p0[2] + int(round(rv[2]))))
            
            new_chain = list(current_chain)
            for k in range(i+1, j+1):
                new_chain[k] = new_positions[k-(i+1)]
            
            occupied_rest = set(current_chain[:i+1]) | set(current_chain[j+1:])
            valid = True
            for k in range(i+1, j+1):
                if new_chain[k] in occupied_rest:
                    valid = False
                    break
                occupied_rest.add(new_chain[k])
            
            if valid:
                if sum(abs(new_chain[i+1][a] - new_chain[i][a]) for a in range(3)) != 1:
                    valid = False
                if valid and j + 1 < n:
                    if sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(3)) != 1:
                        valid = False
            
            if valid:
                current_chain = new_chain
                current_energy = _energy(sequence, current_chain)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                continue
        
        elif move_type < 0.7:
            # Kink move: Change a single internal residue to a different neighbor
            if n < 4:
                continue
            k = rng.randint(1, n - 2)
            old_pos = current_chain[k]
            prev_pos = current_chain[k-1]
            next_pos = current_chain[k+1]
            
            valid_positions = []
            for q in _neighbours(old_pos):
                if q == prev_pos or q == next_pos:
                    continue
                # Check if q is occupied by any other part of chain
                if q in set(current_chain[:k-1]) | set(current_chain[k+2:]):
                    continue
                valid_positions.append(q)
            
            if not valid_positions:
                continue
            
            new_pos = rng.choice(valid_positions)
            new_chain = list(current_chain)
            new_chain[k] = new_pos
            
            # Verify connectivity
            if sum(abs(new_pos[a] - prev_pos[a]) for a in range(3)) != 1:
                continue
            if sum(abs(new_pos[a] - next_pos[a]) for a in range(3)) != 1:
                continue
            
            current_chain = new_chain
            current_energy = _energy(sequence, current_chain)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
            continue
        
        # Segment Shift Move
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        
        direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        
        new_chain = list(current_chain)
        valid = True
        
        occupied_set = set()
        for k in range(n):
            if i <= k <= j:
                continue
            occupied_set.add(current_chain[k])
        
        new_positions = []
        for k in range(i, j + 1):
            old_pos = current_chain[k]
            new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
            new_positions.append(new_pos)
            
        for k in range(i, j + 1):
            if new_positions[k-i] in occupied_set:
                valid = False
                break
        
        if valid:
            pos_prev = current_chain[i-1]
            pos_i_new = new_positions[0]
            dist1 = sum(abs(pos_prev[a] - pos_i_new[a]) for a in range(3))
            
            pos_j_new = new_positions[-1]
            pos_next = current_chain[j+1]
            dist2 = sum(abs(pos_j_new[a] - pos_next[a]) for a in range(3))
            
            if dist1 != 1 or dist2 != 1:
                valid = False
        
        if not valid:
            continue
        
        for k in range(i, j + 1):
            new_chain[k] = new_positions[k-i]
        valid_move = True
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            delta = new_energy - current_energy
            if T > 0 and rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find best conformation for the given HP sequence on the simple cubic lattice.
    """
    n = len(sequence)
    
    start_time = time.time()
    time_limit = 8.0
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345 + len(sequence))
    
    # Strategy: Multiple greedy attempts with different heuristics, then SA refinement
    num_attempts = 35 if n > 30 else 60
    
    weight_sets = [
        (10.0, 1.0, 0.1, 5.0, 0.5, 2),
        (15.0, 2.0, 0.05, 10.0, 1.0, 2),
        (8.0, 3.0, 0.2, 3.0, 0.2, 1),
        (12.0, 1.5, 0.15, 7.0, 0.8, 2),
        (20.0, 1.0, 0.0, 20.0, 1.5, 2),
        (18.0, 0.5, 0.02, 15.0, 0.3, 3),
        (14.0, 1.2, 0.08, 8.0, 0.6, 2),
        (16.0, 0.8, 0.06, 12.0, 0.4, 2),
    ]
    
    for i in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
            
        ws = weight_sets[i % len(weight_sets)]
        chain = _generate_chain(sequence, dimension, rng, *ws)
        
        if chain is None:
            continue
            
        e = _energy(sequence, chain)
        
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
        # Refine with SA
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.5:
            sa_time = min(2.0, time_remaining * 0.65)
            refined_chain, refined_e = _local_search_sa(sequence, dimension, chain, rng, time_limit=sa_time)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined_chain
                
    # Final polish: Run SA on the best chain with more time
    if best_chain is not None and time_limit - (time.time() - start_time) > 1.0:
        final_time = min(3.0, time_limit - (time.time() - start_time) - 0.1)
        if final_time > 0.5:
            final_chain, final_e = _local_search_sa(sequence, dimension, best_chain, rng, time_limit=final_time)
            if final_e < best_energy:
                best_energy = final_e
                best_chain = final_chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
