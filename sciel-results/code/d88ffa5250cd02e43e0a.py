import random
import math
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_walk(sequence, dimension, rng):
    N = len(sequence)
    # Start with a straight line
    chain = [(i, 0, 0) for i in range(N)]
    # Scramble with random pivots to get a more compact initial state
    for _ in range(N * 5):
        if N < 3: break
        pivot_idx = rng.randint(1, N - 2)
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        
        # Perform pivot
        # Extract segment
        seg = chain[pivot_idx:]
        pivot_coord = chain[pivot_idx]
        rel_coords = [(p[0]-pivot_coord[0], p[1]-pivot_coord[1], p[2]-pivot_coord[2]) for p in seg]
        
        def rotate_point(p, ax, d):
            x, y, z = p
            if ax == 0:
                return (x, -z*d, y*d) if d==1 else (x, z*d, -y*d) # Wait, rotation logic needs to be precise
                # 90 deg around X: (x,y,z) -> (x, -z, y) for d=1, (x, z, -y) for d=-1
                # Let's use the explicit formulas from before
            # Re-using the logic from before to be safe
            if ax == 0:
                if d == 1: return (x, -z, y)
                else: return (x, z, -y)
            elif ax == 1:
                if d == 1: return (z, y, -x)
                else: return (-z, y, x)
            else:
                if d == 1: return (y, -x, z)
                else: return (-y, x, z)

        new_rel_coords = [rotate_point(rc, axis, direction) for rc in rel_coords]
        new_seg = [(pivot_coord[0]+rc[0], pivot_coord[1]+rc[1], pivot_coord[2]+rc[2]) for rc in new_rel_coords]
        new_chain = chain[:pivot_idx] + new_seg
        
        # Check validity
        fixed_part = set(chain[:pivot_idx])
        valid = True
        for p in new_seg:
            if p in fixed_part:
                valid = False
                break
        
        if valid:
            chain = new_chain
            
    return chain

def _pivot_move(chain, pivot_idx, axis, direction, rng):
    N = len(chain)
    if pivot_idx <= 0 or pivot_idx >= N - 1:
        return None
    
    seg = chain[pivot_idx:]
    pivot_coord = chain[pivot_idx]
    rel_coords = [(p[0]-pivot_coord[0], p[1]-pivot_coord[1], p[2]-pivot_coord[2]) for p in seg]
    
    def rotate_point(p, ax, d):
        x, y, z = p
        if ax == 0:
            if d == 1: return (x, -z, y)
            else: return (x, z, -y)
        elif ax == 1:
            if d == 1: return (z, y, -x)
            else: return (-z, y, x)
        else:
            if d == 1: return (y, -x, z)
            else: return (-y, x, z)

    new_rel_coords = [rotate_point(rc, axis, direction) for rc in rel_coords]
    new_seg = [(pivot_coord[0]+rc[0], pivot_coord[1]+rc[1], pivot_coord[2]+rc[2]) for rc in new_rel_coords]
    new_chain = chain[:pivot_idx] + new_seg
    
    fixed_part = set(chain[:pivot_idx])
    for p in new_seg:
        if p in fixed_part:
            return None
            
    return new_chain

def _c_terminal_move(chain, rng):
    if len(chain) < 2:
        return None
    last = chain[-1]
    neighbors = list(_neighbours(last))
    rng.shuffle(neighbors)
    
    occupied = set(chain[:-1])
    
    for nb in neighbors:
        if nb not in occupied:
            return chain[:-1] + [nb]
    return None

def _sa_fold(sequence, dimension, time_limit):
    N = len(sequence)
    if N <= 1:
        return [(0,0,0)]
    
    rng = random.Random(hash(sequence) % (2**32))
    
    chain = _initial_walk(sequence, dimension, rng)
    current_energy = _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    T = max(1.0, N / 10.0)
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < time_limit:
        if rng.random() < 0.8:
            pivot_idx = rng.randint(1, N - 2)
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            new_chain = _pivot_move(chain, pivot_idx, axis, direction, rng)
        else:
            new_chain = _c_terminal_move(chain, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 0.01 and rng.random() < math.exp(-delta_e / T)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        T *= 0.999
        step_count += 1
        
    return best_chain

def fold(sequence, dimension):
    N = len(sequence)
    if N < 20:
        time_limit = 2.0
    elif N < 50:
        time_limit = 4.0
    else:
        time_limit = 8.0
        
    chain = _sa_fold(sequence, dimension, time_limit)
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
