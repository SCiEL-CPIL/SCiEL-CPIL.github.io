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
                
                score = new_energy - 0.5 * free_count + rng.random() * 0.1
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

def _local_search(sequence, chain, dimension, rng, time_limit):
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_energy -= 1
                    
    best_energy = current_energy
    best_chain = chain.copy()
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        k = rng.randint(1, n-1)
        pivot_site = chain[k-1]
        segment = chain[k:]
        
        for p in segment:
            del occupied[p]
            
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
        
        new_segment = []
        for p in segment:
            v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
            nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
            nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
            nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
            new_segment.append((pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2))
        
        valid = True
        for p in new_segment:
            if p in occupied:
                valid = False
                break
                
        if not valid:
            for i, p in enumerate(segment):
                occupied[p] = k + i
            continue
            
        old_contacts = 0
        new_contacts = 0
        for i, p in enumerate(segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in occupied:
                        nb_idx = occupied[nb]
                        if is_h[nb_idx]:
                            old_contacts += 1
                            
        for i, p in enumerate(new_segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in occupied:
                        nb_idx = occupied[nb]
                        if is_h[nb_idx]:
                            new_contacts += 1
                            
        delta_contacts = new_contacts - old_contacts
        delta_energy = -delta_contacts
        T = 0.5
        if delta_energy < 0:
            accept = True
        else:
            prob = math.exp(-delta_energy / T)
            accept = rng.random() < prob
            
        if accept:
            for i, p in enumerate(new_segment):
                occupied[p] = k + i
                chain[k + i] = p
            current_energy += delta_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain.copy()
        else:
            for i, p in enumerate(segment):
                occupied[p] = k + i
                
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    PER_CALL_TIME_LIMIT = 8.0
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
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

    initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=50)
    if initial_chain is None or len(initial_chain) != n:
        initial_chain = [(i, 0, 0) for i in range(n)]
        
    init_energy = _calc_energy(sequence, initial_chain)
    best_energy = init_energy
    best_chain = initial_chain.copy()
    
    remaining_time = deadline - time.time()
    if remaining_time > 0.5:
        refined_chain = _local_search(sequence, initial_chain, dimension, rng, remaining_time - 0.1)
        ref_energy = _calc_energy(sequence, refined_chain)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined_chain.copy()
            
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        rng2 = random.Random(99999 + len(sequence))
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=50)
        if initial_chain2 and len(initial_chain2) == n:
            rem2 = deadline - time.time() - 0.1
            if rem2 > 0.5:
                refined_chain2 = _local_search(sequence, initial_chain2, dimension, rng2, rem2)
                ref_energy2 = _calc_energy(sequence, refined_chain2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined_chain2.copy()
            else:
                e2 = _calc_energy(sequence, initial_chain2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = initial_chain2.copy()

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
