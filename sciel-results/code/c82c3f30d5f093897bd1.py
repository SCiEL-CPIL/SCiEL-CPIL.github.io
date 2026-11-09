"""HP lattice protein folding using biased growth + simulated annealing with incremental energy."""
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation."""
    n = len(chain)
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _incremental_energy_delta(sequence, chain, i, old_pos, new_pos):
    """Compute the change in energy when moving residue i from old_pos to new_pos."""
    # Build pos_to_idx for quick lookup
    n = len(chain)
    pos_to_idx = {}
    for j, p in enumerate(chain):
        pos_to_idx[p] = j
    
    # Count H-H contacts involving residue i before and after the move
    def count_contacts_involving(idx, pos):
        count = 0
        if sequence[idx] != 'H':
            return 0
        x, y = pos
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j != idx and sequence[j] == 'H':
                # Only count if j > idx to avoid double counting
                if j > idx:
                    count += 1
        return count
    
    # Before move
    before = count_contacts_involving(i, old_pos)
    
    # After move: update pos_to_idx temporarily
    del pos_to_idx[old_pos]
    pos_to_idx[new_pos] = i
    
    after = count_contacts_involving(i, new_pos)
    
    # Restore
    del pos_to_idx[new_pos]
    pos_to_idx[old_pos] = i
    
    return after - before

def _biased_grow(sequence, dimension, rng):
    """Grow a chain using biased random walk favoring compact conformations."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        # Bias towards moving towards the centroid for compactness
        if i > 2:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            
            # Score each candidate by distance to centroid (prefer closer)
            def score(site):
                return (site[0] - cx) ** 2 + (site[1] - cy) ** 2
            
            # Weighted selection: prefer sites closer to centroid
            weights = [1.0 / (1.0 + score(s)) for s in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen = candidates[0]
            for s, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = s
                    break
        else:
            chosen = rng.choice(candidates)
        
        chain.append(chosen)
        occupied.add(chosen)
    
    return chain

def _reverse_move(chain, rng):
    """Reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None, None, None
    
    start = rng.randint(0, n - 3)
    end = rng.randint(start + 1, n - 1)
    
    # Check if reversal is valid (no self-intersection)
    new_chain = chain[:start] + chain[start:end+1][::-1] + chain[end+1:]
    
    if len(set(new_chain)) != len(new_chain):
        return None, None, None
    
    return new_chain, start, end

def _pivot_move(chain, rng):
    """Pivot a segment around its first residue."""
    n = len(chain)
    if n < 5:
        return None
    
    start = rng.randint(0, n - 4)
    end = rng.randint(start + 2, n - 1)
    
    pivot = chain[start]
    segment = chain[start+1:end+1]
    
    # Try all 3 possible 90-degree rotations
    for rotation in range(3):
        rotated = []
        for pos in segment:
            dx = pos[0] - pivot[0]
            dy = pos[1] - pivot[1]
            # Rotate 90 degrees clockwise each time
            new_dx, new_dy = dy, -dx
            rotated.append((pivot[0] + new_dx, pivot[1] + new_dy))
        
        new_chain = chain[:start+1] + rotated + chain[end+1:]
        
        # Check validity
        if len(set(new_chain)) != len(new_chain):
            continue
        
        # Check unit steps
        valid = True
        for i in range(len(new_chain) - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        
        if valid:
            return new_chain
    
    return None

def _anneal(sequence, dimension, rng, time_budget):
    """Run simulated annealing with multiple restarts."""
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget and _remaining_time() > 5.0:
        # Generate initial chain using biased growth
        chain = _biased_grow(sequence, dimension, rng)
        if chain is None:
            # Fallback to simple random walk
            chain = [(0, 0)]
            occupied = {(0, 0)}
            success = True
            for i in range(1, len(sequence)):
                last = chain[-1]
                candidates = [s for s in _neighbours(last) if s not in occupied]
                if not candidates:
                    success = False
                    break
                site = rng.choice(candidates)
                chain.append(site)
                occupied.add(site)
            if not success:
                continue
        
        current_energy = _compute_energy(sequence, chain)
        
        if best_energy == 0 or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
        
        # Annealing parameters
        start_temp = 1.0
        end_temp = 0.01
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - start_time > time_budget or _remaining_time() < 5.0:
                break
            
            progress = step / max_steps
            temperature = start_temp * (end_temp / start_temp) ** progress
            
            # Choose move type
            move_type = rng.randint(0, 1)
            new_chain = None
            
            if move_type == 0:
                new_chain, s, e = _reverse_move(chain, rng)
                if new_chain is None:
                    continue
            else:
                new_chain = _pivot_move(chain, rng)
                if new_chain is None:
                    continue
            
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.001)):
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using biased growth + simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Allocate time based on sequence length
    if n <= 30:
        time_per_run = min(3.0, _remaining_time() / 15.0)
        max_runs = 20
    elif n <= 60:
        time_per_run = min(4.0, _remaining_time() / 10.0)
        max_runs = 15
    else:
        time_per_run = min(5.0, _remaining_time() / 8.0)
        max_runs = 10
    
    for attempt in range(max_runs):
        if _remaining_time() < 10.0:
            break
        
        chain, energy = _anneal(sequence, dimension, rng, time_per_run)
        if chain is not None and (best_energy == 0 or energy < best_energy):
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: simple straight chain
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
