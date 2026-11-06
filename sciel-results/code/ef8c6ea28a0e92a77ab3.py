import random
import time
import math

# Precompute 90-degree rotation matrices for 3D
_ROT_MATRICES = [
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[0,-1,0],[1,0,0],[0,0,1]],
    [[0,1,0],[-1,0,0],[0,0,1]],
]

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _neighbours_list(point):
    return [(point[0]+dx, point[1]+dy, point[2]+dz) for dx, dy, dz in _NB_DIRS]

def _beam_search_growth(sequence, dimension, rng, beam_width=50):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start = (0, 0, 0)
    beams = [([(0,0,0)], 0, {(0,0,0): 0})]
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        candidates = []
        for chain, energy, occupied in beams:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if is_h[i]:
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if is_h[j]:
                                new_energy -= 1
                
                free_count = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_count += 1
                
                score = new_energy - 0.3 * free_count + rng.random() * 0.15
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_chain, new_energy, new_occupied))
        
        if not candidates:
            break
            
        candidates.sort(key=lambda x: x[0])
        beams = [(c[1], c[2], c[3]) for c in candidates[:beam_width]]
        
    if not beams:
        return [(i, 0, 0) for i in range(n)]
        
    best = min(beams, key=lambda x: x[1])
    return best[0]

def _compute_energy(sequence, chain):
    n = len(chain)
    if n < 2:
        return 0
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    e = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    e -= 1
    return e

def _apply_pivot_move(chain, k, mat, pivot_site, occupied):
    """Apply a pivot move at position k. Returns (new_segment, valid, delta_energy)"""
    n = len(chain)
    new_segment = []
    for i in range(k, n):
        p = chain[i]
        v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
        nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
        nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
        nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
        new_segment.append((pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2))
    
    # Check validity
    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
    
    return new_segment, valid

def _compute_delta_energy(sequence, chain, new_segment, k, occupied, is_h):
    """Compute delta energy for a move that replaces chain[k:] with new_segment"""
    n = len(chain)
    # Old contacts: between segment and fixed part (indices < k)
    old_delta = 0
    for i in range(k, n):
        if is_h[i]:
            for nb in _neighbours(chain[i]):
                j = occupied.get(nb)
                if j is not None and j < k and is_h[j]:
                    old_delta += 1
    
    # New contacts: between new_segment and fixed part
    new_delta = 0
    for i in range(len(new_segment)):
        idx = k + i
        if is_h[idx]:
            for nb in _neighbours(new_segment[i]):
                j = occupied.get(nb)
                if j is not None and j < k and is_h[j]:
                    new_delta += 1
    
    delta_contacts = new_delta - old_delta
    delta_energy = -delta_contacts
    return delta_energy

def _local_search_multi(sequence, chain, dimension, rng, time_limit):
    """Multi-neighborhood local search: pivot, crank, kink, and end moves"""
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain.copy()
    start_time = time.time()
    
    T_start = 2.0
    T_end = 0.05
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        T = T_start * (T_end / T_start) ** frac
        
        # Choose move type: 0=pivot, 1=crank, 2=kink, 3=end
        move_type = rng.choices([0, 1, 2, 3], weights=[40, 25, 20, 15])[0]
        
        if move_type == 0:  # Pivot move
            k = rng.randint(1, n-1)
            if n - k > n // 2:
                continue
            pivot_site = chain[k-1]
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
            
            # Remove segment from occupied
            for i in range(k, n):
                del occupied[chain[i]]
            
            new_segment, valid = _apply_pivot_move(chain, k, mat, pivot_site, occupied)
            
            if not valid:
                for i in range(k, n):
                    occupied[chain[i]] = i
                continue
            
            delta_energy = _compute_delta_energy(sequence, chain, new_segment, k, occupied, is_h)
            
            if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
                for i in range(len(new_segment)):
                    occupied[new_segment[i]] = k + i
                    chain[k + i] = new_segment[i]
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                for i in range(k, n):
                    occupied[chain[i]] = i
                    
        elif move_type == 1:  # Crank move: rotate 4-residue segment
            if n < 5:
                continue
            k = rng.randint(1, n-4)
            pivot_site = chain[k-1]
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
            
            # Only rotate residues k to k+3
            seg_len = 4
            for i in range(k, k + seg_len):
                del occupied[chain[i]]
            
            new_segment = []
            valid = True
            for i in range(k, k + seg_len):
                p = chain[i]
                v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
                nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
                nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
                nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
                np_ = (pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2)
                if np_ in occupied:
                    valid = False
                    break
                new_segment.append(np_)
            
            if not valid:
                for i in range(k, k + seg_len):
                    occupied[chain[i]] = i
                continue
            
            # Compute delta energy
            old_delta = 0
            for i in range(k, k + seg_len):
                if is_h[i]:
                    for nb in _neighbours(chain[i]):
                        j = occupied.get(nb)
                        if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                            old_delta += 1
            
            new_delta = 0
            for i in range(seg_len):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(new_segment[i]):
                        j = occupied.get(nb)
                        if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                            new_delta += 1
            
            delta_energy = -(new_delta - old_delta)
            
            if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
                for i in range(seg_len):
                    occupied[new_segment[i]] = k + i
                    chain[k + i] = new_segment[i]
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                for i in range(k, k + seg_len):
                    occupied[chain[i]] = i
                    
        elif move_type == 2:  # Kink move: rotate 2-residue segment
            if n < 3:
                continue
            k = rng.randint(1, n-2)
            pivot_site = chain[k-1]
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
            
            seg_len = 2
            for i in range(k, k + seg_len):
                del occupied[chain[i]]
            
            new_segment = []
            valid = True
            for i in range(k, k + seg_len):
                p = chain[i]
                v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
                nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
                nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
                nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
                np_ = (pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2)
                if np_ in occupied:
                    valid = False
                    break
                new_segment.append(np_)
            
            if not valid:
                for i in range(k, k + seg_len):
                    occupied[chain[i]] = i
                continue
            
            old_delta = 0
            for i in range(k, k + seg_len):
                if is_h[i]:
                    for nb in _neighbours(chain[i]):
                        j = occupied.get(nb)
                        if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                            old_delta += 1
            
            new_delta = 0
            for i in range(seg_len):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(new_segment[i]):
                        j = occupied.get(nb)
                        if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                            new_delta += 1
            
            delta_energy = -(new_delta - old_delta)
            
            if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
                for i in range(seg_len):
                    occupied[new_segment[i]] = k + i
                    chain[k + i] = new_segment[i]
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                for i in range(k, k + seg_len):
                    occupied[chain[i]] = i
                    
        elif move_type == 3:  # End move: move last 1-3 residues
            if n < 3:
                continue
            seg_len = rng.randint(1, min(3, n-1))
            k = n - seg_len
            pivot_site = chain[k-1]
            
            # Try to find a new position for the segment
            found = False
            for _ in range(10):
                # Try a random direction from pivot
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
                
                for i in range(k, n):
                    del occupied[chain[i]]
                
                new_segment = []
                valid = True
                for i in range(k, n):
                    p = chain[i]
                    v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
                    nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
                    nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
                    nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
                    np_ = (pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2)
                    if np_ in occupied:
                        valid = False
                        break
                    new_segment.append(np_)
                
                if valid:
                    old_delta = 0
                    for i in range(k, n):
                        if is_h[i]:
                            for nb in _neighbours(chain[i]):
                                j = occupied.get(nb)
                                if j is not None and j < k and is_h[j]:
                                    old_delta += 1
                    
                    new_delta = 0
                    for i in range(len(new_segment)):
                        idx = k + i
                        if is_h[idx]:
                            for nb in _neighbours(new_segment[i]):
                                j = occupied.get(nb)
                                if j is not None and j < k and is_h[j]:
                                    new_delta += 1
                    
                    delta_energy = -(new_delta - old_delta)
                    
                    if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
                        for i in range(len(new_segment)):
                            occupied[new_segment[i]] = k + i
                            chain[k + i] = new_segment[i]
                        current_energy += delta_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = chain.copy()
                        found = True
                        break
                    else:
                        for i in range(k, n):
                            occupied[chain[i]] = i
                else:
                    for i in range(k, n):
                        occupied[chain[i]] = i
            
            if not found:
                for i in range(k, n):
                    occupied[chain[i]] = i
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    PER_CALL_TIME_LIMIT = 11.5
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    def _calc_energy(seq, ch):
        return _compute_energy(seq, ch)

    # Use smaller beam width for longer chains to save time
    bw = 30 if n > 60 else 50
    initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=bw)
    if initial_chain is None or len(initial_chain) != n:
        initial_chain = [(i, 0, 0) for i in range(n)]
        
    init_energy = _calc_energy(sequence, initial_chain)
    best_energy = init_energy
    best_chain = initial_chain.copy()
    
    remaining_time = deadline - time.time()
    if remaining_time > 0.5:
        refined_chain = _local_search_multi(sequence, initial_chain, dimension, rng, remaining_time - 0.2)
        ref_energy = _calc_energy(sequence, refined_chain)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined_chain.copy()
            
    # Second restart with different seed if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        rng2 = random.Random(99999 + len(sequence))
        bw2 = 25 if n > 60 else 40
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=bw2)
        if initial_chain2 and len(initial_chain2) == n:
            rem2 = deadline - time.time() - 0.2
            if rem2 > 0.5:
                refined_chain2 = _local_search_multi(sequence, initial_chain2, dimension, rng2, rem2)
                ref_energy2 = _calc_energy(sequence, refined_chain2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined_chain2.copy()
            else:
                e2 = _calc_energy(sequence, initial_chain2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = initial_chain2.copy()
    
    # Third restart if still time
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        rng3 = random.Random(55555 + len(sequence))
        bw3 = 20 if n > 60 else 35
        initial_chain3 = _beam_search_growth(sequence, dimension, rng3, beam_width=bw3)
        if initial_chain3 and len(initial_chain3) == n:
            rem3 = deadline - time.time() - 0.2
            if rem3 > 0.5:
                refined_chain3 = _local_search_multi(sequence, initial_chain3, dimension, rng3, rem3)
                ref_energy3 = _calc_energy(sequence, refined_chain3)
                if ref_energy3 < best_energy:
                    best_energy = ref_energy3
                    best_chain = refined_chain3.copy()
            else:
                e3 = _calc_energy(sequence, initial_chain3)
                if e3 < best_energy:
                    best_energy = e3
                    best_chain = initial_chain3.copy()

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Final validation: ensure self-avoidance and unit steps
    seen = set()
    for i, p in enumerate(best_chain):
        if p in seen:
            # Fallback to linear chain
            best_chain = [(i, 0, 0) for i in range(n)]
            break
        seen.add(p)
        if i > 0:
            prev = best_chain[i-1]
            d = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if d != 1:
                best_chain = [(i, 0, 0) for i in range(n)]
                break
    
    return [list(p) for p in best_chain]
