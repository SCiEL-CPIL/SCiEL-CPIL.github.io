"""HP lattice protein folding using pivot-move simulated annealing with greedy construction."""
import time
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy_single(sequence, chain, i, new_pos, occupied):
    """Compute energy change if residue i moves to new_pos."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0
    delta = 0
    if sequence[i] == 'H':
        # Remove old contacts
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                delta += 1
        # Add new contacts
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                delta -= 1
    return -delta

def _greedy_grow(sequence, rng, compactness=0.3, contact_weight=10.0):
    """Grow a chain greedily, preferring positions that maximize H-H contacts and compactness."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            
            # Contact score: how many H-H contacts does placing here create?
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += contact_weight
            
            # Compactness: prefer positions close to centroid
            if len(chain) > 3:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy)
                score -= dist * compactness
            
            # Avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            if free == 0 and i < n - 1:
                score -= 100.0
            elif free == 1 and i < n - 3:
                score -= 5.0
            
            # Small random perturbation
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        # Sometimes explore
        if rng.random() < 0.15 and len(candidates) > 1:
            idx = min(rng.randint(1, 2), len(candidates) - 1)
            best_site = candidates[idx][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _try_pivot_move(chain, pivot_index, angle, occupied):
    """Try a pivot move at pivot_index by rotating the suffix by angle (in 90-degree increments).
    Returns (new_chain, new_occupied) if valid, else (None, None)."""
    n = len(chain)
    if pivot_index < 1 or pivot_index >= n - 1:
        return None, None
    
    # Extract the suffix to rotate
    suffix = chain[pivot_index:]
    pivot_point = chain[pivot_index - 1]
    
    # Rotate suffix around pivot_point
    # angle is in radians, but we only use 90-degree turns
    # For 90-degree rotation: (x,y) -> (y, -x) for clockwise, (-y, x) for counter-clockwise
    
    # The suffix positions relative to pivot_point
    rel_suffix = [(p[0] - pivot_point[0], p[1] - pivot_point[1]) for p in suffix]
    
    # Apply rotation
    if angle == 1:  # 90 degrees clockwise
        rotated = [(y, -x) for (x, y) in rel_suffix]
    elif angle == 2:  # 180 degrees
        rotated = [(-x, -y) for (x, y) in rel_suffix]
    elif angle == 3:  # 270 degrees clockwise (or 90 counter-clockwise)
        rotated = [(-y, x) for (x, y) in rel_suffix]
    else:
        return None, None
    
    # New absolute positions
    new_suffix = [(pivot_point[0] + x, pivot_point[1] + y) for (x, y) in rotated]
    
    # Check that the first point of new suffix is adjacent to pivot_point
    if abs(new_suffix[0][0] - pivot_point[0]) + abs(new_suffix[0][1] - pivot_point[1]) != 1:
        return None, None
    
    # Build new chain
    new_chain = chain[:pivot_index] + new_suffix
    
    # Check self-avoidance
    new_occupied = {}
    for i, p in enumerate(new_chain):
        if p in new_occupied:
            return None, None
        new_occupied[p] = i
    
    # Verify unit steps
    for i in range(1, n):
        if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
            return None, None
    
    return new_chain, new_occupied

def _pivot_search(sequence, chain, rng, max_iter=5000, start_temp=1.5, end_temp=0.05):
    """Simulated annealing with pivot moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    temp = start_temp
    cooling = math.log(start_temp / end_temp) / max_iter if max_iter > 0 else 0
    
    for iteration in range(max_iter):
        temp *= math.exp(-cooling)
        if temp < end_temp:
            break
        
        # Choose pivot index (weighted towards middle)
        pivot = rng.randint(1, n - 2)
        angle = rng.randint(1, 3)
        
        new_chain, new_occupied = _try_pivot_move(current_chain, pivot, angle, occupied)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            current_chain = new_chain
            current_energy = new_energy
            occupied = new_occupied
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_iter=200):
    """Greedy hill-climbing polish with single-residue moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        for i in range(1, n - 1):
            old_pos = current_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                delta_e = _delta_energy_single(sequence, current_chain, i, new_pos, occupied)
                
                if delta_e < 0:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta_e
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 195.0
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate multiple initial chains using greedy growth
    num_initial = 30
    for i in range(num_initial):
        if time.time() > deadline - 120.0:
            break
        
        # Vary parameters slightly
        compactness = 0.2 + rng.random() * 0.4
        contact_weight = 8.0 + rng.random() * 4.0
        
        chain = _greedy_grow(sequence, rng, compactness, contact_weight)
        
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Fallback: simple straight line if no chain found
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Pivot-move simulated annealing on the best chain
    if time.time() < deadline - 60.0:
        sa_chain, sa_energy = _pivot_search(
            sequence, best_chain, rng,
            max_iter=3000, start_temp=1.5, end_temp=0.05
        )
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Phase 3: Greedy polish
    if time.time() < deadline - 30.0:
        polish_chain, polish_energy = _greedy_polish(sequence, best_chain, max_iter=300)
        if polish_energy < best_energy:
            best_chain = polish_chain
            best_energy = polish_energy
    
    # Phase 4: Another round of SA from the polished chain
    if time.time() < deadline - 15.0:
        sa2_chain, sa2_energy = _pivot_search(
            sequence, best_chain, rng,
            max_iter=2000, start_temp=1.0, end_temp=0.03
        )
        if sa2_energy < best_energy:
            best_chain = sa2_chain
            best_energy = sa2_energy
    
    # Phase 5: Final polish
    if time.time() < deadline - 5.0:
        final_chain, final_energy = _greedy_polish(sequence, best_chain, max_iter=200)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy
    
    return [list(p) for p in best_chain]
