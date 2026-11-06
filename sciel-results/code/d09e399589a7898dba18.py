import random
import time
import math
import numpy as np

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
_NB_ARRAY = np.array(_NB_DIRS, dtype=np.int32)

_ROT_MATRICES = [
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[0,-1,0],[1,0,0],[0,0,1]],
    [[0,1,0],[-1,0,0],[0,0,1]],
]
_ROT_ARRAYS = [np.array(m, dtype=np.int32) for m in _ROT_MATRICES]

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _neighbours_list(point):
    return [(point[0]+dx, point[1]+dy, point[2]+dz) for dx, dy, dz in _NB_DIRS]

def _compute_energy(sequence, chain):
    n = len(chain)
    if n < 2:
        return 0
    is_h = np.array([1 if s == 'H' else 0 for s in sequence], dtype=np.int32)
    occupied = {p: i for i, p in enumerate(chain)}
    e = 0
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(chain[i]):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    e -= 1
    return e

def _beam_search_growth(sequence, dimension, rng, beam_width=50, time_limit=None):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start_time = time.time()
    is_h = [1 if s == 'H' else 0 for s in sequence]
    # Remaining H count for heuristic
    remaining_h = [0] * n
    count = 0
    for i in range(n-1, -1, -1):
        count += is_h[i]
        remaining_h[i] = count
    
    # Beams: list of (chain_list, energy, occupied_dict, score)
    beams = [([(0,0,0)], 0, {(0,0,0): 0}, 0.0)]
    
    for i in range(1, n):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        candidates = []
        for chain, energy, occupied, _ in beams:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and is_h[j]:
                            new_energy -= 1
                
                # Free space count
                free_count = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_count += 1
                
                # A* heuristic: energy + remaining H potential - free space penalty
                # Higher remaining H means more potential contacts to gain
                # Lower free space means more compact
                rem_h = remaining_h[i]
                score = new_energy - 0.3 * free_count + 0.5 * rem_h + rng.random() * 0.1
                
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_chain, new_energy, new_occupied))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        # Keep top beam_width, but ensure diversity
        beams = [(c[1], c[2], c[3], c[0]) for c in candidates[:beam_width]]
    
    if not beams:
        return [(i, 0, 0) for i in range(n)]
    
    best = min(beams, key=lambda x: x[1])
    return best[0]

def _local_search(sequence, chain, dimension, rng, time_limit):
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = np.array([1 if s == 'H' else 0 for s in sequence], dtype=np.int32)
    chain_arr = np.array(chain, dtype=np.int32)
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Compute initial energy
    current_energy = 0
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(tuple(chain_arr[i])):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_energy -= 1
                    
    best_energy = current_energy
    best_chain = chain.copy()
    start_time = time.time()
    
    T_start = 2.0
    T_end = 0.1
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        T = T_start * (T_end / T_start) ** frac
        
        k = rng.randint(1, n-1)
        segment_len = n - k
        
        if segment_len > n // 2:
            continue
            
        # Remove segment from occupied
        seg_positions = chain[k:]
        for i in range(k, n):
            del occupied[chain[i]]
            
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        mat = _ROT_ARRAYS[axis*2 + (0 if direction == 1 else 1)]
        
        pivot = np.array(chain[k-1], dtype=np.int32)
        seg_vec = chain_arr[k:] - pivot
        new_seg_vec = (mat @ seg_vec.T).T
        new_segment = (pivot + new_seg_vec).tolist()
        
        # Check validity
        valid = True
        for p in new_segment:
            if tuple(p) in occupied:
                valid = False
                break
        
        if not valid:
            for i in range(k, n):
                occupied[chain[i]] = i
            continue
            
        # Compute delta energy
        old_delta = 0
        for i in range(k, n):
            if is_h[i]:
                for nb in _neighbours(tuple(chain[i])):
                    j = occupied.get(nb)
                    if j is not None and j < k and is_h[j]:
                        old_delta += 1
                        
        new_delta = 0
        for i in range(len(new_segment)):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(tuple(new_segment[i])):
                    j = occupied.get(nb)
                    if j is not None and j < k and is_h[j]:
                        new_delta += 1
                        
        delta_contacts = new_delta - old_delta
        delta_energy = -delta_contacts
        
        if delta_energy < 0:
            accept = True
        else:
            prob = math.exp(-delta_energy / T)
            accept = rng.random() < prob
            
        if accept:
            for i in range(len(new_segment)):
                occupied[tuple(new_segment[i])] = k + i
                chain[k + i] = new_segment[i]
                chain_arr[k + i] = new_segment[i]
            current_energy += delta_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain.copy()
        else:
            for i in range(k, n):
                occupied[chain[i]] = i
                chain_arr[k + i] = chain[i]
                
    return best_chain

def _single_residue_polish(sequence, chain, time_limit):
    """Final polish with single-residue shifts"""
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    
    current_energy = 0
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(chain[i]):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_energy -= 1
                    
    best_energy = current_energy
    best_chain = chain.copy()
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, n-1)
        if i == 0 or i == n-1:
            continue
        
        # Try moving residue i to a neighboring site
        for site in _neighbours(chain[i]):
            if site in occupied:
                continue
            # Check if moving i to site breaks continuity
            prev = chain[i-1]
            nxt = chain[i+1]
            d_prev = abs(site[0]-prev[0]) + abs(site[1]-prev[1]) + abs(site[2]-prev[2])
            d_next = abs(site[0]-nxt[0]) + abs(site[1]-nxt[1]) + abs(site[2]-nxt[2])
            if d_prev != 1 or d_next != 1:
                continue
            
            # Compute delta
            old_contacts = 0
            if is_h[i]:
                for nb in _neighbours(chain[i]):
                    j = occupied.get(nb)
                    if j is not None and j != i-1 and j != i+1 and is_h[j]:
                        old_contacts += 1
            
            new_contacts = 0
            if is_h[i]:
                for nb in _neighbours(site):
                    j = occupied.get(nb)
                    if j is not None and j != i-1 and j != i+1 and is_h[j]:
                        new_contacts += 1
            
            delta = new_contacts - old_contacts
            delta_energy = -delta
            
            if delta_energy <= 0:
                del occupied[chain[i]]
                occupied[site] = i
                chain[i] = site
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
                break
                
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    GLOBAL_DEADLINE = time.time() + 310.0  # 20s safety margin
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    def _calc_energy(seq, ch):
        occ = {p: i for i, p in enumerate(ch)}
        e = 0
        is_h = [1 if s == 'H' else 0 for s in seq]
        for i, p in enumerate(ch):
            if is_h[i]:
                for q in _neighbours(p):
                    j = occ.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        e -= 1
        return e
    
    # Adaptive beam width based on chain length
    if n <= 48:
        bw_primary = 80
        bw_secondary = 60
        bw_tertiary = 50
    elif n <= 80:
        bw_primary = 50
        bw_secondary = 40
        bw_tertiary = 30
    else:
        bw_primary = 35
        bw_secondary = 25
        bw_tertiary = 20
    
    # Primary restart
    time_left = GLOBAL_DEADLINE - time.time()
    if time_left > 2.0:
        beam_time = min(time_left * 0.3, 5.0)
        initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=bw_primary, time_limit=beam_time)
        if initial_chain and len(initial_chain) == n:
            init_energy = _calc_energy(sequence, initial_chain)
            best_energy = init_energy
            best_chain = initial_chain.copy()
            
            # Local search
            time_left = GLOBAL_DEADLINE - time.time()
            if time_left > 1.0:
                ls_time = min(time_left * 0.4, 8.0)
                refined = _local_search(sequence, initial_chain, dimension, rng, ls_time)
                ref_energy = _calc_energy(sequence, refined)
                if ref_energy < best_energy:
                    best_energy = ref_energy
                    best_chain = refined.copy()
    
    # Secondary restart
    time_left = GLOBAL_DEADLINE - time.time()
    if time_left > 3.0:
        rng2 = random.Random(99999 + len(sequence))
        beam_time = min(time_left * 0.25, 4.0)
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=bw_secondary, time_limit=beam_time)
        if initial_chain2 and len(initial_chain2) == n:
            e2 = _calc_energy(sequence, initial_chain2)
            if e2 < best_energy:
                best_energy = e2
                best_chain = initial_chain2.copy()
            
            time_left = GLOBAL_DEADLINE - time.time()
            if time_left > 1.0:
                ls_time = min(time_left * 0.35, 6.0)
                refined2 = _local_search(sequence, initial_chain2, dimension, rng2, ls_time)
                ref_energy2 = _calc_energy(sequence, refined2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined2.copy()
    
    # Tertiary restart
    time_left = GLOBAL_DEADLINE - time.time()
    if time_left > 3.0:
        rng3 = random.Random(55555 + len(sequence))
        beam_time = min(time_left * 0.2, 3.0)
        initial_chain3 = _beam_search_growth(sequence, dimension, rng3, beam_width=bw_tertiary, time_limit=beam_time)
        if initial_chain3 and len(initial_chain3) == n:
            e3 = _calc_energy(sequence, initial_chain3)
            if e3 < best_energy:
                best_energy = e3
                best_chain = initial_chain3.copy()
            
            time_left = GLOBAL_DEADLINE - time.time()
            if time_left > 1.0:
                ls_time = min(time_left * 0.3, 5.0)
                refined3 = _local_search(sequence, initial_chain3, dimension, rng3, ls_time)
                ref_energy3 = _calc_energy(sequence, refined3)
                if ref_energy3 < best_energy:
                    best_energy = ref_energy3
                    best_chain = refined3.copy()
    
    # Final polish with single-residue moves
    time_left = GLOBAL_DEADLINE - time.time()
    if time_left > 0.5 and best_chain is not None:
        rng4 = random.Random(77777 + len(sequence))
        polish_time = min(time_left * 0.5, 3.0)
        polished = _single_residue_polish(sequence, best_chain, polish_time)
        pol_energy = _calc_energy(sequence, polished)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_chain = polished.copy()
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Final validation
    seen = set()
    valid = True
    for i, p in enumerate(best_chain):
        if p in seen:
            valid = False
            break
        seen.add(p)
        if i > 0:
            prev = best_chain[i-1]
            d = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if d != 1:
                valid = False
                break
    
    if not valid:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
