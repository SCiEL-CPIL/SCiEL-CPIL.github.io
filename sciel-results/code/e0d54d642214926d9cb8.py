"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search_simple(sequence, chain, rng, max_steps=1000, deadline_check=None):
    """Local search with simple moves: pivot, 90-degree rotation."""
    current = chain[:]
    current_energy = _energy(sequence, current)
    occupied = {p: i for i, p in enumerate(current)}
    
    for _ in range(max_steps):
        if deadline_check and deadline_check():
            break
            
        n = len(current)
        move_type = rng.choice([0, 1, 2])  # 0: pivot, 1: rotate, 2: shift
        
        if move_type == 0:  # Pivot
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            # Try all 5 rotations for the segment from i to end
            segment = current[i:]
            # Generate rotations
            rotations = []
            # 0 deg (current)
            rotations.append(segment)
            
            # 90 deg clockwise around axis 0
            seg_rot = []
            for p in segment:
                new_p = [p[0], -p[2], p[1]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 180 deg
            seg_rot = []
            for p in segment:
                new_p = [-p[0], -p[1], -p[2]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 270 deg clockwise
            seg_rot = []
            for p in segment:
                new_p = [p[0], p[2], -p[1]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 90 deg around axis 1
            seg_rot = []
            for p in segment:
                new_p = [p[2], p[1], -p[0]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # Pick a random rotation
            rot_idx = rng.randint(0, 4)
            new_segment = rotations[rot_idx]
            
            # Check for collisions
            valid = True
            for j, p in enumerate(new_segment):
                idx = i + j
                if idx < n - 1:
                    # Check against all previous residues
                    for k in range(idx):
                        if p == current[k]:
                            valid = False
                            break
                else:
                    for k in range(idx):
                        if p == current[k]:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                new_chain = current[:i] + new_segment
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current = new_chain
                    current_energy = new_energy
                    occupied = {p: i for i, p in enumerate(current)}
        
        elif move_type == 1:  # 90-degree rotation of a sub-segment
            if n < 4:
                continue
            length = rng.randint(2, min(5, n - 1))
            i = rng.randint(1, n - length)
            segment = current[i:i+length]
            
            # Try rotating the segment
            rotations = []
            # 90 deg around axis 0
            seg_rot = []
            for p in segment:
                new_p = [p[0], -p[2], p[1]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 180 deg
            seg_rot = []
            for p in segment:
                new_p = [-p[0], -p[1], -p[2]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 270 deg
            seg_rot = []
            for p in segment:
                new_p = [p[0], p[2], -p[1]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            # 90 deg around axis 1
            seg_rot = []
            for p in segment:
                new_p = [p[2], p[1], -p[0]]
                seg_rot.append(tuple(new_p))
            rotations.append(seg_rot)
            
            rot_idx = rng.randint(0, 3)
            new_segment = rotations[rot_idx]
            
            valid = True
            for j, p in enumerate(new_segment):
                idx = i + j
                for k in range(n):
                    if k != idx and p == current[k]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_chain = current[:i] + new_segment + current[i+length:]
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current = new_chain
                    current_energy = new_energy
                    occupied = {p: i for i, p in enumerate(current)}
        
        else:  # Shift (move a single residue)
            if n < 2:
                continue
            i = rng.randint(1, n - 1)
            p = current[i]
            for new_p in _neighbours(p):
                if new_p not in [current[k] for k in range(n) if k != i]:
                    new_chain = current[:]
                    new_chain[i] = new_p
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current = new_chain
                        current_energy = new_energy
                        occupied = {p: i for i, p in enumerate(current)}
                        break
    
    return current, current_energy

def _simulated_annealing(sequence, chain, rng, max_iterations=5000, deadline_check=None):
    """Simulated annealing with diverse moves."""
    current = chain[:]
    current_energy = _energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    
    t_start = 10.0
    t_end = 0.1
    iterations = max_iterations
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, iterations - 1))
    t = t_start
    
    for _ in range(iterations):
        if deadline_check and deadline_check():
            break
            
        # Generate a random move
        move_type = rng.choice([0, 1, 2])
        n = len(current)
        new_chain = current[:]
        
        if move_type == 0:  # Pivot
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            segment = current[i:]
            rotations = []
            rotations.append(segment)
            seg_rot = [tuple([p[0], -p[2], p[1]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([-p[0], -p[1], -p[2]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([p[0], p[2], -p[1]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([p[2], p[1], -p[0]]) for p in segment]
            rotations.append(seg_rot)
            
            rot_idx = rng.randint(0, 4)
            new_segment = rotations[rot_idx]
            
            valid = True
            for j, p in enumerate(new_segment):
                idx = i + j
                for k in range(idx):
                    if p == current[k]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_chain = current[:i] + new_segment
        
        elif move_type == 1:  # Rotate sub-segment
            if n < 4:
                continue
            length = rng.randint(2, min(5, n - 1))
            i = rng.randint(1, n - length)
            segment = current[i:i+length]
            rotations = []
            seg_rot = [tuple([p[0], -p[2], p[1]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([-p[0], -p[1], -p[2]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([p[0], p[2], -p[1]]) for p in segment]
            rotations.append(seg_rot)
            seg_rot = [tuple([p[2], p[1], -p[0]]) for p in segment]
            rotations.append(seg_rot)
            
            rot_idx = rng.randint(0, 3)
            new_segment = rotations[rot_idx]
            
            valid = True
            for j, p in enumerate(new_segment):
                idx = i + j
                for k in range(n):
                    if k != idx and p == current[k]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_chain = current[:i] + new_segment + current[i+length:]
        
        else:  # Shift
            if n < 2:
                continue
            i = rng.randint(1, n - 1)
            p = current[i]
            for new_p in _neighbours(p):
                if new_p not in [current[k] for k in range(n) if k != i]:
                    new_chain = current[:]
                    new_chain[i] = new_p
                    break
        
        # Calculate energy difference
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / t):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy
        
        t *= cooling_rate
    
    return best, best_energy

def fold(sequence, dimension):
    """Best conformation using multi-start simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 1
    
    # Time budget: allocate time based on sequence length
    # Shorter sequences get more restarts, longer get fewer
    n = len(sequence)
    if n <= 48:
        max_restarts = 10
        max_sa_iters = 2000
    elif n <= 100:
        max_restarts = 5
        max_sa_iters = 1500
    else:
        max_restarts = 3
        max_sa_iters = 1000
    
    for restart in range(max_restarts):
        if _time_exceeded():
            break
        
        # Generate initial chain
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Run simulated annealing
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, rng, 
                                                   max_iterations=max_sa_iters, 
                                                   deadline_check=_time_exceeded)
        
        if sa_energy < best_energy:
            best = sa_chain
            best_energy = sa_energy
        
        # If we've found a good energy, reduce restarts
        if best_energy < -n // 2:
            break
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
