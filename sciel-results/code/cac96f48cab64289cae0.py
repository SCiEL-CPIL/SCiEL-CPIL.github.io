import time
import random

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _beam_search(sequence, beam_width, max_time):
    N = len(sequence)
    if N <= 1:
        return [(0, [(0, 0)])]
    
    start = (0, 0)
    initial_states = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        p1 = (dx, dy)
        initial_states.append((0, [start, p1], {start: 0, p1: 1}, p1))
    
    current_beam = initial_states
    start_time = time.time()
    
    for i in range(2, N):
        next_beam = []
        for energy, chain, occupied, last_p in current_beam:
            for q in _neighbours(last_p):
                if q in occupied:
                    continue
                
                gain = 0
                if sequence[i] == 'H':
                    for r in _neighbours(q):
                        j = occupied.get(r)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                new_energy = energy - gain
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                next_beam.append((new_energy, new_chain, new_occupied, q))
        
        if not next_beam:
            return [(e, c) for e, c, _, _ in current_beam]
            
        next_beam.sort(key=lambda x: x[0])
        current_beam = next_beam[:beam_width]
        
        if time.time() - start_time > max_time:
            break
            
    return [(e, c) for e, c, _, _ in current_beam]

def _calculate_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, max_time):
    N = len(chain)
    if N <= 2:
        return chain, _calculate_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.999
    rng = random.Random(42)
    
    while time.time() - start_time < max_time:
        i = rng.randint(0, N - 1)
        p_old = current_chain[i]
        
        for q in _neighbours(p_old):
            if q == p_old:
                continue
            
            if q in occupied and occupied[q] != i:
                continue
            
            if i > 0:
                p_prev = current_chain[i - 1]
                if abs(p_prev[0] - q[0]) + abs(p_prev[1] - q[1]) != 1:
                    continue
            if i < N - 1:
                p_next = current_chain[i + 1]
                if abs(p_next[0] - q[0]) + abs(p_next[1] - q[1]) != 1:
                    continue
            
            delta_e = 0
            
            if sequence[i] == 'H':
                for r in _neighbours(p_old):
                    j = occupied.get(r)
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                        delta_e += 1
            
            if sequence[i] == 'H':
                for r in _neighbours(q):
                    j = occupied.get(r)
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                        delta_e -= 1
            
            if delta_e < 0:
                current_chain[i] = q
                occupied.pop(p_old)
                occupied[q] = i
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            else:
                if rng.random() < pow(2.71828, -delta_e / T):
                    current_chain[i] = q
                    occupied.pop(p_old)
                    occupied[q] = i
                    current_energy += delta_e
            
            break
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 200.0
    
    N = len(sequence)
    if N == 0:
        return []
    if N == 1:
        return [[0, 0]]
    
    beam_width = 100
    beam_results = _beam_search(sequence, beam_width, max_time=10.0)
    
    if not beam_results:
        return [[i, 0] for i in range(N)]
    
    candidates = []
    seen = set()
    for e, c in beam_results:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            candidates.append((e, c))
        if len(candidates) >= 5:
            break
    
    best_chain = candidates[0][1]
    best_energy = candidates[0][0]
    
    elapsed = time.time() - start_time
    time_remaining = max(0, deadline - elapsed)
    
    if time_remaining > 1.0:
        time_per_candidate = time_remaining / len(candidates)
        
        for e, c in candidates:
            if time.time() > deadline:
                break
            
            refined_chain, refined_energy = _local_search(sequence, c, max_time=time_per_candidate)
            
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
    
    return [list(p) for p in best_chain]
