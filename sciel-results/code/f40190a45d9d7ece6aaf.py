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
    
    # Beam items: (energy, chain, occupied)
    beams = [(0, [(0,0,0)], {(0,0,0): 0})]
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        candidates = []
        for energy, chain, occupied in beams:
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
                
                # Simple score: energy + small random tie-breaker
                score = new_energy + rng.random() * 0.01
                
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_energy, new_chain, new_occupied))
        
        if not candidates:
            break
            
        # Sort by score (ascending) - lower is better
        candidates.sort(key=lambda x: x[0])
        # Keep top beam_width
        beams = [(c[1], c[2], c[3]) for c in candidates[:beam_width]]
        
    if not beams:
        return [(i, 0, 0) for i in range(n)]
        
    # Return the one with the lowest energy
    best = min(beams, key=lambda x: x[0])
    return best[1]

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

def _local_search_steepest(sequence, chain, dimension, rng, time_limit):
    """
    Steepest Descent Local Search with Random Kicks.
    Tries multiple candidate moves per iteration and picks the best improving one.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    current_chain = chain.copy()
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain.copy()
    best_energy = current_energy
    start_time = time.time()
    
    # Parameters
    MAX_STUCK = 8  # Number of failed improvement attempts before a kick
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        kick_needed = False
        stuck_count += 1
        
        if stuck_count > MAX_STUCK:
            kick_needed = True
            stuck_count = 0
            
        if not kick_needed:
            # Try up to 8 candidate moves, pick the best improving one
            best_delta = 0
            best_move = None
            for attempt in range(8):
                max_seg_len = min(6, n - 1)
                weights = [40, 25, 15, 8, 6, 6][:max_seg_len]
                total_weight = sum(weights)
                r = rng.random() * total_weight
                cumulative = 0
                seg_len = 1
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        seg_len = i + 1
                        break
                
                k = rng.randint(1, n - seg_len)
                pivot_site = current_chain[k-1]
                
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
                
                # Remove segment from occupied
                old_segment = current_chain[k:k+seg_len]
                for p in old_segment:
                    del occupied[p]
                    
                # Apply rotation
                new_segment = []
                valid = True
                for i in range(seg_len):
                    p = old_segment[i]
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
                    for i, p in enumerate(old_segment):
                        occupied[p] = k + i
                    continue
                    
                # Calculate delta energy
                old_contacts = 0
                for i in range(seg_len):
                    idx = k + i
                    if is_h[idx]:
                        for nb in _neighbours(old_segment[i]):
                            j = occupied.get(nb)
                            if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                                old_contacts += 1
                                
                new_contacts = 0
                for i in range(seg_len):
                    idx = k + i
                    if is_h[idx]:
                        for nb in _neighbours(new_segment[i]):
                            j = occupied.get(nb)
                            if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                                new_contacts += 1
                                
                delta_energy = -(new_contacts - old_contacts)
                
                if delta_energy < best_delta:
                    best_delta = delta_energy
                    best_move = (k, seg_len, old_segment, new_segment, delta_energy)
                
                # Restore
                for i, p in enumerate(old_segment):
                    occupied[p] = k + i
                
                if best_delta < -2:
                    break  # Good enough improvement, stop searching
            
            if best_move is not None:
                k, seg_len, old_segment, new_segment, delta_energy = best_move
                for p in old_segment:
                    del occupied[p]
                for i, p in enumerate(new_segment):
                    occupied[p] = k + i
                    current_chain[k + i] = p
                current_energy += delta_energy
                stuck_count = 0
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain.copy()
            else:
                stuck_count += 1
        else:
            # Kick: Apply a random valid move, even if worse
            max_seg_len = min(6, n - 1)
            seg_len = rng.randint(1, max_seg_len)
            k = rng.randint(1, n - seg_len)
            pivot_site = current_chain[k-1]
            
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
            
            old_segment = current_chain[k:k+seg_len]
            for p in old_segment:
                del occupied[p]
                
            new_segment = []
            valid = True
            for i in range(seg_len):
                p = old_segment[i]
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
                for i, p in enumerate(new_segment):
                    occupied[p] = k + i
                    current_chain[k + i] = p
                
                # Recompute energy for kick
                current_energy = _compute_energy(sequence, current_chain)
            else:
                for i, p in enumerate(old_segment):
                    occupied[p] = k + i

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    # Total budget is 330s for 26 sequences. ~12.7s per sequence.
    # Use 11.0s per call to be safe with overhead.
    PER_CALL_TIME_LIMIT = 11.0
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Beam Search Initialization
    bw = 20 if n > 60 else 60
    initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=bw)
    if initial_chain is None or len(initial_chain) != n:
        initial_chain = [(i, 0, 0) for i in range(n)]
        
    init_energy = _compute_energy(sequence, initial_chain)
    best_energy = init_energy
    best_chain = initial_chain.copy()
    
    # Local Search Refinement - main phase
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        ls_time = remaining_time - 0.3
        refined_chain = _local_search_steepest(sequence, initial_chain, dimension, rng, ls_time)
        ref_energy = _compute_energy(sequence, refined_chain)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined_chain.copy()
            
    # Second restart if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        rng2 = random.Random(99999 + len(sequence))
        bw2 = 15 if n > 60 else 40
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=bw2)
        if initial_chain2 and len(initial_chain2) == n:
            rem2 = deadline - time.time() - 0.2
            if rem2 > 0.5:
                refined_chain2 = _local_search_steepest(sequence, initial_chain2, dimension, rng2, rem2)
                ref_energy2 = _compute_energy(sequence, refined_chain2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined_chain2.copy()
            else:
                e2 = _compute_energy(sequence, initial_chain2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = initial_chain2.copy()
    
    # Third restart if still time
    remaining_time = deadline - time.time()
    if remaining_time > 1.5:
        rng3 = random.Random(55555 + len(sequence))
        bw3 = 10 if n > 60 else 30
        initial_chain3 = _beam_search_growth(sequence, dimension, rng3, beam_width=bw3)
        if initial_chain3 and len(initial_chain3) == n:
            rem3 = deadline - time.time() - 0.15
            if rem3 > 0.4:
                refined_chain3 = _local_search_steepest(sequence, initial_chain3, dimension, rng3, rem3)
                ref_energy3 = _compute_energy(sequence, refined_chain3)
                if ref_energy3 < best_energy:
                    best_energy = ref_energy3
                    best_chain = refined_chain3.copy()
            else:
                e3 = _compute_energy(sequence, initial_chain3)
                if e3 < best_energy:
                    best_energy = e3
                    best_chain = initial_chain3.copy()

    # Final polish: try a few more moves on the best chain if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 0.3 and best_chain is not None:
        rng4 = random.Random(77777 + len(sequence))
        polished = _local_search_steepest(sequence, best_chain, dimension, rng4, remaining_time - 0.1)
        pol_energy = _compute_energy(sequence, polished)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_chain = polished.copy()

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Final validation
    seen = set()
    for i, p in enumerate(best_chain):
        if p in seen:
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
