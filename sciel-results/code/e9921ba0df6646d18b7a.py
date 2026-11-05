import random
import time
import math

def _compute_energy(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _generate_random_saw(length, dimension):
    if length <= 0:
        return []
    chain = [tuple([0]*dimension)]
    visited = {chain[0]}
    for _ in range(length - 1):
        last = chain[-1]
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(last)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in visited:
                    neighbors.append(nb)
        if not neighbors:
            return None
        next_pos = random.choice(neighbors)
        chain.append(next_pos)
        visited.add(next_pos)
    return chain

def _crankshaft_move(chain, i, j, dimension):
    """
    Rotate chain[i..j] around the bond chain[i-1]-chain[i].
    Returns new_chain or None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1 or j < i or j >= n:
        return None
    
    # Determine axis
    p_prev = chain[i-1]
    p_curr = chain[i]
    
    axis = -1
    for d in range(dimension):
        if p_prev[d] != p_curr[d]:
            axis = d
            break
            
    if axis == -1:
        return None
        
    # The other two axes
    other_axes = [d for d in range(dimension) if d != axis]
    a1, a2 = other_axes
    
    # Rotation in the plane (a1, a2)
    # We have 4 possible 90-degree rotations?
    # Actually, rotating a vector (x, y) by 90 deg gives (-y, x).
    # Rotating by -90 deg gives (y, -x).
    # Rotating by 180 deg gives (-x, -y).
    # We should try all 3 non-zero rotations (90, 180, 270).
    
    head = chain[:i]
    tail = chain[j+1:]
    segment = chain[i:j+1]
    
    visited = set(head)
    visited.update(set(tail))
    
    # Try 3 rotations
    for angle in [90, 180, 270]:
        new_segment = []
        valid = True
        for p in segment:
            # Vector from p_curr to p
            v1 = p[a1] - p_curr[a1]
            v2 = p[a2] - p_curr[a2]
            
            if angle == 90:
                nv1 = -v2
                nv2 = v1
            elif angle == 180:
                nv1 = -v1
                nv2 = -v2
            else: # 270
                nv1 = v2
                nv2 = -v1
                
            new_p = list(p_curr)
            new_p[a1] += nv1
            new_p[a2] += nv2
            # Axis coordinate remains same as p_curr? 
            # No, p_curr is the pivot point for the rotation of the segment relative to p_curr?
            # The bond is p_prev -> p_curr. The rotation is around this bond.
            # So p_curr stays fixed? No, p_curr is part of the segment.
            # The axis of rotation passes through p_curr and p_prev.
            # So p_curr is on the axis, its coordinates (a1, a2) relative to itself are 0.
            # So for p_curr, v1=0, v2=0, so new_p = p_curr. Correct.
            
            new_p = tuple(new_p)
            
            if new_p in visited:
                valid = False
                break
            new_segment.append(new_p)
            visited.add(new_p)
            
        if valid:
            return head + new_segment + tail
            
    return None

def _pivot_move(chain, k, dimension):
    """
    Rotate chain[k:] around chain[k-1].
    """
    n = len(chain)
    if k <= 0 or k >= n:
        return None
        
    p_prev = chain[k-1]
    p_curr = chain[k]
    
    axis = -1
    for d in range(dimension):
        if p_prev[d] != p_curr[d]:
            axis = d
            break
            
    if axis == -1:
        return None
        
    other_axes = [d for d in range(dimension) if d != axis]
    a1, a2 = other_axes
    
    head = chain[:k]
    segment = chain[k:]
    
    visited = set(head)
    
    for angle in [90, 180, 270]:
        new_segment = []
        valid = True
        for p in segment:
            v1 = p[a1] - p_curr[a1]
            v2 = p[a2] - p_curr[a2]
            
            if angle == 90:
                nv1 = -v2
                nv2 = v1
            elif angle == 180:
                nv1 = -v1
                nv2 = -v2
            else:
                nv1 = v2
                nv2 = -v1
                
            new_p = list(p_curr)
            new_p[a1] += nv1
            new_p[a2] += nv2
            new_p = tuple(new_p)
            
            if new_p in visited:
                valid = False
                break
            new_segment.append(new_p)
            visited.add(new_p)
            
        if valid:
            return head + new_segment
            
    return None

def _simulated_annealing(sequence, dimension, time_limit):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)], 0

    chain = None
    for _ in range(20):
        chain = _generate_random_saw(n, dimension)
        if chain:
            break
    if not chain:
        chain = [(i, 0, 0) for i in range(n)]
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T_start = 3.0
    T_end = 0.01
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Choose move: 50% Crankshaft, 50% Pivot
        if random.random() < 0.5:
            # Crankshaft
            i = random.randint(1, n - 2)
            j = random.randint(i, n - 1)
            new_chain = _crankshaft_move(chain, i, j, dimension)
        else:
            # Pivot
            k = random.randint(1, n - 1)
            new_chain = _pivot_move(chain, k, dimension)
            
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        elapsed = time.time() - start_time
        cooling_const = math.log(T_start / T_end) / (0.5 * time_limit)
        T = T_start * math.exp(-cooling_const * elapsed)
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    best_chain_global = None
    best_energy_global = 0
    
    while time.time() - start_time < time_budget * 0.95:
        remaining = time_budget * 0.95 - (time.time() - start_time)
        if remaining <= 0:
            break
        episode_time = min(1.0, remaining)
        
        chain, energy = _simulated_annealing(sequence, dimension, episode_time)
        
        if best_energy_global == 0 or energy < best_energy_global:
            best_energy_global = energy
            best_chain_global = chain
            
    if best_chain_global is None:
        chain = _generate_random_saw(n, dimension)
        if not chain:
            chain = [(i, 0, 0) for i in range(n)]
        best_chain_global = chain
        
    return [list(p) for p in best_chain_global]
