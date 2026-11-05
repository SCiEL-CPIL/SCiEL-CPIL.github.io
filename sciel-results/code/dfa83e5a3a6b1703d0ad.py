import time
import random

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _compute_energy(chain, sequence):
    """Compute the energy of a full chain."""
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    energy = 0
    for i in range(n):
        if sequence[i] == 'H':
            for nb in _neighbours(chain[i]):
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                        energy -= 1
    return energy // 2

def _local_search(chain, sequence, max_steps=5000, time_limit=10):
    """Improve a chain using local search."""
    start = time.time()
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(best_chain, sequence)
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        
        best_improvement = None
        best_new_energy = best_energy
        
        # Try moving one residue
        for i in range(1, n-1):
            prev_pos = best_chain[i-1]
            next_pos = best_chain[i+1]
            cur_pos = best_chain[i]
            
            for nb in _neighbours(cur_pos):
                if nb == prev_pos or nb == next_pos:
                    continue
                occupied = set(best_chain)
                occupied.remove(cur_pos)
                if nb in occupied:
                    continue
                new_chain = best_chain[:]
                new_chain[i] = nb
                valid = True
                for k in range(n-1):
                    a, b = new_chain[k], new_chain[k+1]
                    if abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) != 1:
                        valid = False
                        break
                if not valid:
                    continue
                e = _compute_energy(new_chain, sequence)
                if e < best_new_energy:
                    best_new_energy = e
                    best_improvement = new_chain
        
        if best_improvement is not None:
            best_chain = best_improvement
            best_energy = best_new_energy
        else:
            if random.random() < 0.1:
                i = random.randint(1, n-2)
                cur_pos = best_chain[i]
                options = [nb for nb in _neighbours(cur_pos) 
                          if nb != best_chain[i-1] and nb != best_chain[i+1]]
                if options:
                    nb = random.choice(options)
                    occupied = set(best_chain)
                    occupied.remove(cur_pos)
                    if nb not in occupied:
                        new_chain = best_chain[:]
                        new_chain[i] = nb
                        valid = True
                        for k in range(n-1):
                            a, b = new_chain[k], new_chain[k+1]
                            if abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) != 1:
                                valid = False
                                break
                        if valid:
                            e = _compute_energy(new_chain, sequence)
                            if e < best_energy or random.random() < 0.05:
                                best_chain = new_chain
                                best_energy = e
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search on the 3D cubic lattice.
    Returns a list of lists of integers representing the coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 280
    
    initial_chain = [(0, 0, 0), (1, 0, 0)]
    initial_occupied = {(0, 0, 0), (1, 0, 0)}
    initial_pos_to_idx = {(0, 0, 0): 0, (1, 0, 0): 1}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_pos_to_idx, initial_energy)]
    
    if n < 50:
        K_base = 400
    elif n < 80:
        K_base = 500
    else:
        K_base = 600
        
    for i in range(2, n):
        new_beam_candidates = []
        residue = sequence[i]
        
        for state in beam:
            chain, occupied, pos_to_idx, energy = state
            last_pos = chain[-1]
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                new_energy = energy
                if residue == 'H':
                    for nb in _neighbours(neighbor):
                        if nb in occupied:
                            j = pos_to_idx[nb]
                            if j != i - 1 and sequence[j] == 'H':
                                new_energy -= 1
                            
                free_space = 0
                for nb in _neighbours(neighbor):
                    if nb not in occupied:
                        free_space += 1
                            
                score = new_energy - 0.05 * free_space
                
                new_beam_candidates.append((score, new_energy, neighbor, chain, occupied, pos_to_idx))
        
        if not new_beam_candidates:
            break
            
        new_beam_candidates.sort(key=lambda x: x[0])
        
        remaining = n - i
        if remaining < 10:
            K = max(100, K_base // 3)
        elif remaining < 20:
            K = max(200, K_base // 2)
        else:
            K = K_base
            
        beam = []
        seen_positions = set()
        for item in new_beam_candidates[:K]:
            score, energy, neighbor, parent_chain, parent_occupied, parent_pos_to_idx = item
            if neighbor in seen_positions:
                continue
            seen_positions.add(neighbor)
            new_chain = parent_chain + [neighbor]
            new_occupied = parent_occupied | {neighbor}
            new_pos_to_idx = parent_pos_to_idx | {neighbor: i}
            beam.append((new_chain, new_occupied, new_pos_to_idx, energy))
            
        if not beam:
            break
            
        if time.time() > deadline:
            break

    if not beam:
        return [[i, 0, 0] for i in range(n)]
    
    best_chain = beam[0][0]
    
    time_left = deadline - time.time()
    if time_left > 5:
        best_chain = _local_search(best_chain, sequence, max_steps=3000, time_limit=min(time_left - 2, 15))
    
    return [list(p) for p in best_chain]
