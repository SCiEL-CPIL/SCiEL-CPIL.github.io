import random
import math
import time

# Global time management
_start_time = None
_deadline = None

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_init(sequence, dimension, rng):
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def fold(sequence, dimension):
    global _start_time
    global _deadline
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0
        
    N = len(sequence)
    if N == 0:
        return []
        
    current_time = time.time()
    time_remaining = _deadline - current_time
    
    if time_remaining <= 1.0:
        # Not enough time, return a valid straight line
        return [[i] + [0]*(dimension-1) for i in range(N)]
        
    # Allocate time
    # Ensure we don't exceed the global deadline
    time_budget = min(15.0, time_remaining * (N / 100.0) + 2.0)
    if time_budget < 1.0:
        time_budget = 1.0
        
    rng = random.Random(12345 + N)
    
    best_chain = None
    best_energy = 0
    
    # Greedy Init
    for _ in range(5):
        chain = _greedy_init(sequence, dimension, rng)
        if chain is not None:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
            # Don't break, try a few more to find a better start
            if e < -10: # Good enough start
                break
                
    if best_chain is None:
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = list(best_chain)
    current_energy = best_energy
    
    T = 1.0
    T_min = 0.01
    
    end_time = time.time() + time_budget
    
    while time.time() < end_time:
        # Fly Move
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        
        if i == 0 and j == N - 1:
            continue
            
        axis = rng.randint(0, dimension - 1)
        step = rng.choice([-1, 1])
        direction = [0] * dimension
        direction[axis] = step
        
        # Check collisions
        static_set = set(current_chain[:i+1])
        static_set.update(current_chain[j+1:])
        
        valid = True
        new_segment = []
        for p in current_chain[i+1:j+1]:
            np_ = (p[0] + direction[0], p[1] + direction[1], p[2] + direction[2])
            if np_ in static_set:
                valid = False
                break
            new_segment.append(np_)
            
        if not valid:
            continue
            
        # Check connections
        if abs(current_chain[i][0] - new_segment[0][0]) + abs(current_chain[i][1] - new_segment[0][1]) + abs(current_chain[i][2] - new_segment[0][2]) != 1:
            continue
            
        if j < N - 1:
            if abs(new_segment[-1][0] - current_chain[j+1][0]) + abs(new_segment[-1][1] - current_chain[j+1][1]) + abs(new_segment[-1][2] - current_chain[j+1][2]) != 1:
                continue
        
        # Apply
        old_segment = current_chain[i+1:j+1]
        current_chain[i+1:j+1] = new_segment
        
        new_energy = _energy(sequence, current_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            current_chain[i+1:j+1] = old_segment
            
        T *= 0.9995
        if T < T_min:
            T = T_min

    return [list(p) for p in best_chain]
