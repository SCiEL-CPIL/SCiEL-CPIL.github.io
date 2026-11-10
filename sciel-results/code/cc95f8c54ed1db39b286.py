"""HP lattice protein folding with multi-replica simulated annealing and 2-opt moves."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            pt = tuple(p)
            for q in _neighbours(pt):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _compact_initial_chain(sequence, dimension, rng):
    """Generate a compact initial self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]}
    x, y, z = 0, 0, 0
    
    # Use a biased direction order to create compact shapes
    for i in range(1, n):
        placed = False
        # Try directions in a compactness-biased order
        dirs = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
        # Shuffle with bias: keep first few in order, shuffle the rest
        rng.shuffle(dirs[3:])
        for dx, dy, dz in dirs:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) not in occupied:
                x, y, z = nx, ny, nz
                chain.append((x, y, z))
                occupied.add((x, y, z))
                placed = True
                break
        if not placed:
            # Fallback: any valid direction
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx, ny, nz) not in occupied:
                    x, y, z = nx, ny, nz
                    chain.append((x, y, z))
                    occupied.add((x, y, z))
                    placed = True
                    break
        if not placed:
            break
    
    return chain

def _sa_fold(sequence, dimension, rng, time_budget):
    """Single replica simulated annealing with 2-opt moves."""
    n = len(sequence)
    if n <= 1:
        return [list(p) for p in [(0, 0, 0)]]
    
    # Initialize with compact chain
    chain = _compact_initial_chain(sequence, dimension, rng)
    if len(chain) < n:
        # Extend if needed
        occupied = set(chain)
        x, y, z = chain[-1]
        for i in range(len(chain), n):
            placed = False
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx, ny, nz) not in occupied:
                    x, y, z = nx, ny, nz
                    chain.append((x, y, z))
                    occupied.add((x, y, z))
                    placed = True
                    break
            if not placed:
                break
        if len(chain) < n:
            # Last resort: straight line
            chain = [(i, 0, 0) for i in range(n)]
    
    current_energy = _energy(sequence, chain)
    best_chain = [tuple(p) for p in chain]
    best_energy = current_energy
    
    t_start = time.time()
    t_min = 0.01
    cooling_rate = 0.9995
    T = max(1.0, abs(current_energy) * 0.3)
    
    while T > t_min:
        if time.time() - t_start > time_budget:
            break
        
        # Pick a 2-opt move
        if n < 4:
            i = rng.randint(1, n - 2)
            j = i + 1
        else:
            max_len = min(n // 2, 15)
            seg_len = rng.randint(2, max_len)
            i = rng.randint(1, n - seg_len - 1)
            j = i + seg_len - 1
        
        # Try the reversal
        new_chain = list(chain)
        new_chain[i:j+1] = list(reversed(new_chain[i:j+1]))
        
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            pt = tuple(p)
            if pt in seen:
                valid = False
                break
            seen.add(pt)
        
        if not valid:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [tuple(p) for p in chain]
        
        T *= cooling_rate
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """Fold a sequence using multi-replica simulated annealing."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Global deadline: 300s total, leave safety margin
    global_deadline = time.time() + 285
    
    # Determine number of replicas based on chain length
    if n <= 48:
        num_replicas = 10
    elif n <= 80:
        num_replicas = 6
    else:
        num_replicas = 4
    
    per_replica_time = 280.0 / num_replicas
    
    best_chain = None
    best_energy = 0
    
    base_seed = 12345 + len(sequence)
    
    for rep in range(num_replicas):
        if time.time() > global_deadline - 5:
            break
        rng = random.Random(base_seed + rep * 7919)
        chain = _sa_fold(sequence, dimension, rng, per_replica_time)
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return best_chain

# EVOLVE-BLOCK-END
