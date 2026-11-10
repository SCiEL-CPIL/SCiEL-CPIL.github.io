"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute energy: -number of H-H non-bond contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # each contact counted twice (once from each H)


def _energy_fast(sequence, chain):
    """Faster energy computation using a set for H positions."""
    n = len(chain)
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(p)
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (0, 1)):  # only check 2 directions to avoid double counting
            if (x + dx, y + dy) in h_positions:
                j = None
                # Check it's not adjacent in chain
                if i > 0 and chain[i-1] == (x+dx, y+dy):
                    continue
                if i < n-1 and chain[i+1] == (x+dx, y+dy):
                    continue
                contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """Grow chain using greedy heuristic with optional randomness."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: number of new H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Also consider if placing here enables future contacts (free space)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10 + free * 0.5 + rng.random() * (1.0 + temperature)
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best (or randomized if temperature > 0)
        if temperature > 0 and len(candidates) > 1:
            # Softmax-like selection
            best_score = max(c[0] for c in candidates)
            weights = [math.exp((c[0] - best_score) / max(temperature, 0.1)) for c in candidates]
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0
            chosen = candidates[-1][1]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c[1]
                    break
        else:
            chosen = max(candidates)[1]
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _kink_flip(chain, i):
    """Try to flip the direction at position i (if possible)."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    if i >= len(chain) - 2:
        return None
    # The segment chain[i-1], chain[i], chain[i+1] forms a corner or straight
    # We can try to reflect chain[i+1] about chain[i]
    x, y = chain[i]
    px, py = chain[i+1]
    new = (2*x - px, 2*y - py)
    # Check if new position is valid (not occupied except by i-1)
    occupied = set(chain)
    occupied.discard(chain[i])  # we're keeping i
    if new in occupied:
        return None
    # Check unit step from i to new
    if abs(new[0] - x) + abs(new[1] - y) != 1:
        return None
    # Build new chain
    new_chain = chain[:i+1] + [new] + chain[i+2:]
    # Verify self-avoidance of the modified part
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    return new_chain


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search using kink flips and short regrows."""
    n = len(chain)
    current_energy = _energy_fast(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0
    temp = 0.5
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        # Randomly pick a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink flip
            i = rng.randint(1, n - 2)
            new_chain = _kink_flip(chain, i)
            if new_chain is None:
                continue
            new_energy = _energy_fast(sequence, new_chain)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Short regrow: remove last k residues and regrow
            k = rng.randint(3, min(8, n - 1))
            if k >= n:
                continue
            prefix = chain[:n-k]
            occupied = {p: i for i, p in enumerate(prefix)}
            suffix = []
            success = True
            for j in range(n-k, n):
                last = prefix[-1] + suffix[-1] if suffix else prefix[-1]
                candidates = []
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    gain = 0
                    if sequence[j] == 'H':
                        for q in _neighbours(site):
                            idx = occupied.get(q)
                            if idx is not None and idx < j - 1 and sequence[idx] == 'H':
                                gain += 1
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    candidates.append((gain * 10 + free * 0.5 + rng.random(), site))
                if not candidates:
                    success = False
                    break
                site = max(candidates)[1]
                suffix.append(site)
                occupied[site] = j
            
            if not success:
                continue
            
            new_chain = prefix + suffix
            new_energy = _energy_fast(sequence, new_chain)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve = 0
                else:
                    no_improve += 1
        
        # Anneal temperature
        temp *= 0.995
        if no_improve > 100:
            temp = max(temp * 0.5, 0.01)
            no_improve = 0
    
    return best_chain, best_energy


def _regrow_sa(sequence, chain, rng, iterations=200):
    """Simulated annealing with regrow moves."""
    n = len(chain)
    current_energy = _energy_fast(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    temp = 1.0
    cooling = 0.99
    
    for it in range(iterations):
        if _time_left() < 5:
            break
        
        # Pick a regrow segment
        seg_len = rng.randint(4, min(12, n - 1))
        if seg_len >= n:
            continue
        start = rng.randint(0, n - seg_len - 1)
        end = start + seg_len
        
        prefix = chain[:start]
        suffix = chain[end:]
        
        occupied = {p: i for i, p in enumerate(prefix)}
        # Also add suffix positions (but they'll be replaced)
        for p in suffix:
            occupied[p] = -1  # mark as occupied but not in prefix
        
        # Regrow the segment
        new_segment = []
        success = True
        last = prefix[-1] if prefix else (0, 0)
        
        for j in range(start, end):
            candidates = []
            for site in _neighbours(last):
                if site in occupied and occupied[site] >= 0:
                    continue
                if site in suffix:
                    # Check if this position is used by suffix (will be removed)
                    # Actually we need to be more careful
                    pass
                gain = 0
                if sequence[j] == 'H':
                    for q in _neighbours(site):
                        idx = occupied.get(q)
                        if idx is not None and idx >= 0 and idx < j - 1 and sequence[idx] == 'H':
                            gain += 1
                free = sum(1 for q in _neighbours(site) 
                          if q not in occupied or occupied[q] < 0)
                candidates.append((gain * 10 + free * 0.5 + rng.random(), site))
            
            if not candidates:
                success = False
                break
            
            site = max(candidates)[1]
            new_segment.append(site)
            occupied[site] = j
            last = site
        
        if not success:
            continue
        
        new_chain = prefix + new_segment + suffix
        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        temp *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find best conformation for the given HP sequence."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # worst possible (no contacts)
    
    # Phase 1: Multiple greedy initializations
    n_greedy = min(50, max(10, 300 // n))
    for i in range(n_greedy):
        if _time_left() < 10:
            break
        temp = 0.1 + 0.3 * (i / max(n_greedy - 1, 1))
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        energy = _energy_fast(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy_fast(sequence, best_chain)
    
    # Phase 2: Local search polish
    if _time_left() > 15:
        polished, polished_energy = _local_search(sequence, best_chain, rng, max_iters=300)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy
    
    # Phase 3: Regrow SA for larger chains
    if _time_left() > 20 and n > 20:
        sa_chain, sa_energy = _regrow_sa(sequence, best_chain, rng, iterations=100)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
        # Polish the SA result
        if _time_left() > 10:
            polished2, polished2_energy = _local_search(sequence, best_chain, rng, max_iters=200)
            if polished2_energy < best_energy:
                best_chain = polished2
                best_energy = polished2_energy
    
    # Additional local search passes if time allows
    while _time_left() > 10:
        polished, polished_energy = _local_search(sequence, best_chain, rng, max_iters=200)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy
        else:
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
