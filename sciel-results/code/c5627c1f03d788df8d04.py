"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    """Generate 6 neighbours of a 3D lattice point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


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


def _delta_energy(sequence, chain, i, new_pos):
    """Compute change in energy when moving residue i to new_pos (must be valid/self-avoiding)."""
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == "H"
    if not is_h:
        return 0  # Moving a P residue doesn't change H-H contacts
    
    # Build position-to-index map excluding residue i
    where = {p: idx for idx, p in enumerate(chain) if idx != i}
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    for q in _neighbours(old_pos):
        j = where.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    for q in _neighbours(new_pos):
        j = where.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            new_contacts += 1
    
    delta = -(new_contacts - old_contacts)
    return delta


def _pivot_move(current_chain, i, axis, sign, rng):
    """Try a pivot move: rotate tail [i+1:] by 90 degrees around axis through pivot.
    Returns (new_chain, delta_energy) or (None, 0) if invalid."""
    n = len(current_chain)
    if i + 1 >= n:
        return None, 0
    
    pivot = current_chain[i]
    tail = current_chain[i+1:]
    
    # Generate rotated tail
    new_tail = []
    for pos in tail:
        rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
        if axis == 0:  # Rotate around x-axis
            if sign == 1:  # +90: (x,y,z) -> (x,-z,y)
                new_rel = (rel[0], -rel[2], rel[1])
            else:  # -90: (x,y,z) -> (x,z,-y)
                new_rel = (rel[0], rel[2], -rel[1])
        elif axis == 1:  # Rotate around y-axis
            if sign == 1:  # +90: (x,y,z) -> (z,y,-x)
                new_rel = (rel[2], rel[1], -rel[0])
            else:  # -90: (x,y,z) -> (-z,y,x)
                new_rel = (-rel[2], rel[1], rel[0])
        else:  # Rotate around z-axis
            if sign == 1:  # +90: (x,y,z) -> (-y,x,z)
                new_rel = (-rel[1], rel[0], rel[2])
            else:  # -90: (x,y,z) -> (y,-x,z)
                new_rel = (rel[1], -rel[0], rel[2])
        new_tail.append((pivot[0] + new_rel[0], pivot[1] + new_rel[1], pivot[2] + new_rel[2]))
    
    # Build new chain
    new_chain = current_chain[:i+1] + new_tail
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None, 0
    
    # Check unit steps (only need to check junction at i and i+1, and within tail which is preserved)
    # Check i to i+1
    diff = abs(new_chain[i+1][0] - new_chain[i][0]) + abs(new_chain[i+1][1] - new_chain[i][1]) + abs(new_chain[i+1][2] - new_chain[i][2])
    if diff != 1:
        return None, 0
    
    return new_chain, 0  # Delta computed by caller


import math


def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with geometric cooling. Returns best chain found."""
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Temperature schedule
    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    margin = 5.0  # seconds before deadline
    
    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain
    
    # Estimate steps per second
    est_steps_per_sec = 5000
    total_steps = int((time_left - margin) * est_steps_per_sec)
    total_steps = max(500, min(total_steps, 50000))  # cap
    
    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0
    
    # Precompute position set for fast collision checking
    pos_set = set(current_chain)
    
    for step in range(total_steps):
        if time.time() >= deadline - margin:
            break
        T *= cooling_factor
        if T < T_min:
            T = T_min
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Flip: move one residue to a random neighbor
            i = rng.randint(1, n - 2)  # not endpoints
            old_pos = current_chain[i]
            
            # Get valid neighbors (unit step from prev and next, not occupied)
            prev = current_chain[i-1]
            nxt = current_chain[i+1]
            
            valid_options = []
            for new_pos in _neighbours(old_pos):
                # Check unit step from prev
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                if d1 != 1:
                    continue
                # Check unit step to next
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d2 != 1:
                    continue
                # Check not occupied (excluding current position)
                if new_pos == old_pos:
                    continue
                if new_pos in pos_set:
                    continue
                valid_options.append(new_pos)
            
            if not valid_options:
                continue
            
            new_pos = rng.choice(valid_options)
            delta = _delta_energy(sequence, current_chain, i, new_pos)
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Update
                pos_set.discard(old_pos)
                pos_set.add(new_pos)
                current_chain[i] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        elif move_type < 0.85:
            # Pivot move: rotate tail by 90 degrees
            i = rng.randint(1, n - 3)
            axis = rng.randint(0, 2)
            sign = rng.choice([1, -1])
            
            pivot = current_chain[i]
            tail = current_chain[i+1:]
            
            # Generate rotated tail
            new_tail = []
            for pos in tail:
                rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
                if axis == 0:
                    if sign == 1:
                        new_rel = (rel[0], -rel[2], rel[1])
                    else:
                        new_rel = (rel[0], rel[2], -rel[1])
                elif axis == 1:
                    if sign == 1:
                        new_rel = (rel[2], rel[1], -rel[0])
                    else:
                        new_rel = (-rel[2], rel[1], rel[0])
                else:
                    if sign == 1:
                        new_rel = (-rel[1], rel[0], rel[2])
                    else:
                        new_rel = (rel[1], -rel[0], rel[2])
                new_tail.append((pivot[0] + new_rel[0], pivot[1] + new_rel[1], pivot[2] + new_rel[2]))
            
            new_chain = current_chain[:i+1] + new_tail
            
            # Check self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Check unit step at junction
            diff = abs(new_chain[i+1][0] - new_chain[i][0]) + abs(new_chain[i+1][1] - new_chain[i][1]) + abs(new_chain[i+1][2] - new_chain[i][2])
            if diff != 1:
                continue
            
            # Compute new energy
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        else:
            # Reversal: reverse a segment of the chain
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            
            new_chain = list(current_chain)
            new_chain[a:b+1] = reversed(new_chain[a:b+1])
            
            # Check self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Check unit steps
            valid = True
            for k in range(n - 1):
                d = abs(new_chain[k+1][0] - new_chain[k][0]) + abs(new_chain[k+1][1] - new_chain[k][1]) + abs(new_chain[k+1][2] - new_chain[k][2])
                if d != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multi-start growth + simulated annealing with deadline guard."""
    deadline = time.time() + 240.0  # 240s budget with safety margin
    
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # 0 = no contacts (unfolded)
    
    n = len(sequence)
    
    # Phase 1: Multiple growth starts
    num_growths = min(30, max(5, int(20.0 / max(1, n) * 300)))  # scale with length
    for g in range(num_growths):
        if time.time() >= deadline - 60:  # leave time for annealing
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Anneal the best chain
    time_left = deadline - time.time()
    if time_left > 10:
        annealed = _anneal(sequence, best, dimension, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy
        
        # If time remains, try another growth and anneal
        if time.time() < deadline - 30:
            chain2 = _grow(sequence, dimension, rng)
            if chain2 is not None:
                e2 = _energy(sequence, chain2)
                if e2 < best_energy:
                    best, best_energy = chain2, e2
                annealed2 = _anneal(sequence, best, dimension, rng, deadline)
                e2a = _energy(sequence, annealed2)
                if e2a < best_energy:
                    best, best_energy = annealed2, e2a
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
