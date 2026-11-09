"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours_2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours_2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _spiral_grow(sequence):
    """Grow a chain in a spiral pattern."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_len = 1
    
    for i in range(1, n):
        placed = False
        for attempt in range(4):
            dx, dy = directions[dir_idx % 4]
            for step in range(step_len):
                new_pos = (chain[-1][0] + dx * (step + 1), chain[-1][1] + dy * (step + 1))
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    placed = True
                    break
            if placed:
                break
            dir_idx += 1
        
        if not placed:
            for dx, dy in directions:
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    placed = True
                    break
        
        if not placed:
            return None
        
        if dir_idx % 4 == 0 and dir_idx > 0:
            step_len = dir_idx // 4 + 1
    
    return chain


def _random_walk_grow(sequence, rng):
    """Grow a chain with random walk, preferring H-H contact opportunities."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours_2d(chain[-1]):
            if site in occupied:
                continue
            score = 0
            if sequence[i] == "H":
                for q in _neighbours_2d(site):
                    if q in occupied:
                        score += 1
            free = sum(1 for q in _neighbours_2d(site) if q not in occupied)
            score += free * 0.1 + rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        # Pick from top candidates with some randomness
        k = min(3, len(candidates))
        _, site = rng.choice(candidates[:k])
        chain.append(site)
        occupied.add(site)
    
    return chain


def _local_search(sequence, chain, rng, max_time):
    """Simulated annealing with efficient energy delta computation."""
    n = len(sequence)
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    # Build lookup
    pos_to_idx = {p: i for i, p in enumerate(current)}
    
    start = time.time()
    t = 5.0
    cool = 0.99
    
    while _remaining_time() > 5.0 and time.time() - start < max_time:
        # Pick move type
        r = rng.random()
        
        if r < 0.5:
            # Single step: move residue i to adjacent empty cell
            i = rng.randint(1, n - 2)
            old_pos = current[i]
            for new_pos in _neighbours_2d(old_pos):
                if new_pos == current[i - 1] or (i + 1 < n and new_pos == current[i + 1]):
                    continue
                if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
                    continue
                
                # Compute delta
                delta = 0
                # Remove old contacts
                if sequence[i] == "H":
                    for q in _neighbours_2d(old_pos):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                # Add new contacts
                if sequence[i] == "H":
                    for q in _neighbours_2d(new_pos):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                
                new_energy = current_energy + delta
                if new_energy < best_energy or rng.random() < (2.718281828) ** (-(new_energy - current_energy) / max(t, 0.1)):
                    # Accept
                    del pos_to_idx[old_pos]
                    pos_to_idx[new_pos] = i
                    current[i] = new_pos
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = list(current)
                    break
        elif r < 0.8:
            # Reversal of segment [i, j]
            if n < 5:
                t *= cool
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            segment = current[i:j + 1]
            rev = list(reversed(segment))
            
            # Check validity: no position in rev (except endpoints) is occupied by outside
            valid = True
            for k, pos in enumerate(rev):
                idx = i + k
                if pos in pos_to_idx:
                    occupant = pos_to_idx[pos]
                    if occupant < i or occupant > j:
                        valid = False
                        break
                    if occupant == i and k != 0:
                        valid = False
                        break
                    if occupant == j and k != j - i:
                        valid = False
                        break
            
            if not valid:
                t *= cool
                continue
            
            # Apply
            current[i:j + 1] = rev
            new_energy = _compute_energy(sequence, current)
            pos_to_idx = {p: idx for idx, p in enumerate(current)}
            
            if new_energy < best_energy or rng.random() < (2.718281828) ** (-(new_energy - current_energy) / max(t, 0.1)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best = list(current)
            else:
                # Revert
                current[i:j + 1] = segment
                pos_to_idx = {p: idx for idx, p in enumerate(current)}
        
        t *= cool
    
    return best, best_energy


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Generate initial conformations
    inits = []
    c = _spiral_grow(sequence)
    if c and len(c) == n:
        inits.append(c)
    
    for _ in range(5):
        if _remaining_time() < 10:
            break
        c = _random_walk_grow(sequence, rng)
        if c and len(c) == n:
            inits.append(c)
    
    for chain in inits:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = 0
    
    # Local search
    time_budget = max(0.5, min(15.0, _remaining_time() / 2))
    best_chain, best_energy = _local_search(sequence, best_chain, rng, time_budget)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
