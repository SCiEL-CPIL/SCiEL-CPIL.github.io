"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))


def _energy(sequence, chain):
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for nb in _neighbours((x, y)):
                j = where.get(nb)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _contact_map(sequence):
    """Pre-compute H-H contact pairs weighted by sequence distance."""
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            dist = j - i
            # Pairs close in sequence are more likely to be spatially close
            # Weight: closer pairs get higher priority
            weight = max(0, 1.0 - dist / (n * 0.6))
            pairs.append((i, j, weight))
    # Sort by weight descending
    pairs.sort(key=lambda x: -x[2])
    return pairs


def _grow_constrained(sequence, rng, contact_pairs, bias=1.0, max_backtrack=0):
    """Growth guided by contact map: prefer placements that satisfy high-priority H-H pairs."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    satisfied = set()  # Set of (i, j) pairs satisfied
    
    for i in range(1, n):
        best_site = None
        best_score = -1e9
        options = []
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            score = 0.0
            
            if sequence[i] == 'H':
                # Count new H-H contacts
                gain = 0
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
                        # Check if this satisfies a high-priority pair
                        pair = (min(i, j), max(i, j))
                        if pair in satisfied:
                            pass  # Already counted
                        else:
                            # Bonus for satisfying high-priority pairs
                            for (pi, pj, w) in contact_pairs[:20]:
                                if (pi, pj) == pair:
                                    score += w * 2.0
                                    break
                
                score += bias * gain
                
                # Also check: does placing this H enable future contacts?
                # Look at how many future H residues are nearby in sequence
                future_h = 0
                for k in range(i + 2, min(n, i + 15)):
                    if sequence[k] == 'H':
                        future_h += 1
                # Slight bonus for being in a region with future H's
                score += 0.1 * future_h
            
            # Compactness: prefer sites closer to centroid
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = abs(site[0] - cx) + abs(site[1] - cy)
            score -= 0.3 * dist
            
            # Free space: prefer sites with more open neighbors (avoid dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            
            # Random perturbation
            score += rng.random() * 0.5
            
            options.append((score, site))
        
        if not options:
            return None
        
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
        
        # Update satisfied pairs
        if sequence[i] == 'H':
            x, y = best_site
            for q in _neighbours((x, y)):
                j = occupied.get(q)
                if j is not None and j < i - 1 and sequence[j] == 'H':
                    satisfied.add((min(i, j), max(i, j)))
    
    return chain


def _grow_random(sequence, rng, bias=1.0):
    """Standard random greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = 0.0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
                score += bias * gain
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = abs(site[0] - cx) + abs(site[1] - cy)
            score -= 0.4 * dist
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            score += rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, rng):
    """Fast pivot move."""
    n = len(chain)
    if n < 4:
        return None
    
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
    
    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
        
        return chain[:segment_start] + rotated
    
    return None


def _slither_move(chain, rng):
    """Slither move: try to move the end."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        return chain[:-1] + [site]
    return None


def _segment_shift(chain, rng):
    """Shift a short segment (2-3 residues) to an adjacent position."""
    n = len(chain)
    if n < 6:
        return None
    
    seg_len = rng.randint(2, 3)
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    segment = chain[start:end]
    rest_before = chain[:start]
    rest_after = chain[end:]
    
    # Try shifting the segment to a neighboring position
    pivot = chain[start - 1]
    fixed_set = set(rest_before)
    
    for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
        new_segment = []
        valid = True
        for p in segment:
            new_p = (p[0] + dx, p[1] + dy)
            if new_p in fixed_set:
                valid = False
                break
            # Also check against rest_after
            if new_p in rest_after:
                valid = False
                break
            new_segment.append(new_p)
        
        if valid:
            # Check that the connection at the boundary is still valid
            if abs(new_segment[0][0] - pivot[0]) + abs(new_segment[0][1] - pivot[1]) != 1:
                continue
            if rest_after and abs(new_segment[-1][0] - rest_after[0][0]) + abs(new_segment[-1][1] - rest_after[0][1]) != 1:
                continue
            return rest_before + new_segment + rest_after
    
    return None


def _local_search(sequence, chain, max_iterations=20000, time_limit=8.0, rng=None):
    """Simulated annealing with multiple move types."""
    if chain is None:
        return None
    
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 6.0
    cooling = 0.997
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break
        
        r = rng.random()
        if r < 0.50:
            new_chain = _pivot_move(current_chain, rng)
        elif r < 0.75:
            new_chain = _slither_move(current_chain, rng)
        elif r < 0.90:
            new_chain = _segment_shift(current_chain, rng)
        else:
            # Long pivot for large sequences
            n = len(current_chain)
            if n > 20:
                pivot_idx = rng.randint(1, max(2, n // 4))
            else:
                pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start < n:
                segment = current_chain[segment_start:]
                fixed_set = set(current_chain[:pivot_idx + 1])
                new_chain = None
                for angle in (90, -90, 180):
                    rotated = []
                    valid = True
                    for p in segment:
                        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                        if angle == 90:
                            ndx, ndy = -dy, dx
                        elif angle == -90:
                            ndx, ndy = dy, -dx
                        else:
                            ndx, ndy = -dx, -dy
                        new_p = (pivot[0] + ndx, pivot[1] + ndy)
                        if new_p in fixed_set:
                            valid = False
                            break
                        rotated.append(new_p)
                    if valid:
                        new_chain = current_chain[:segment_start] + rotated
                        break
            else:
                new_chain = None
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (temperature > 0.5 and rng.random() < math.exp(-delta / temperature)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain


def _polish(sequence, chain, max_iterations=10000, time_limit=5.0, rng=None):
    """Hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    
    if rng is None:
        rng = random.Random(42)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    stall = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break
        
        improved = False
        
        # Try multiple random moves
        for _ in range(30):
            r = rng.random()
            if r < 0.6:
                new_chain = _pivot_move(current_chain, rng)
            else:
                new_chain = _slither_move(current_chain, rng)
            
            if new_chain is None:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
        
        if not improved:
            stall += 1
            if stall > 40:
                break
        else:
            stall = 0
    
    return best_chain


def _spiral_fold(sequence):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (1, 0)]
    
    chain = [(0, 0)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_per_dir = 1
    
    while len(chain) < n:
        dx, dy = directions[dir_idx]
        for _ in range(steps_per_dir):
            if len(chain) >= n:
                break
            lx, ly = chain[-1]
            chain.append((lx + dx, ly + dy))
        
        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 2:
            steps_per_dir += 1
    
    return chain[:n]


def _snake_fold(sequence):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    width = max(1, int(math.sqrt(n)) + 1)
    
    for i in range(1, n):
        row = i // width
        col = i % width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        chain.append((x, y))
    
    return chain


def _validate(chain, seq_len):
    """Check self-avoidance and unit steps."""
    if len(chain) != seq_len:
        return False
    if len(set(chain)) != seq_len:
        return False
    for i in range(1, seq_len):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def fold(sequence, dimension):
    seq_len = len(sequence)
    
    # Adjust budget based on sequence length
    if seq_len <= 30:
        num_grow = 800
        ls_time = 12.0
        polish_time = 6.0
    elif seq_len <= 50:
        num_grow = 550
        ls_time = 10.0
        polish_time = 5.0
    elif seq_len <= 70:
        num_grow = 400
        ls_time = 8.5
        polish_time = 4.5
    else:
        num_grow = 280
        ls_time = 7.0
        polish_time = 4.0
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # Pre-compute contact map for guided growth
    contact_pairs = _contact_map(sequence)
    
    # Pass 0: Deterministic structures
    for init_chain in [_spiral_fold(sequence), _snake_fold(sequence)]:
        if init_chain is not None and _validate(init_chain, seq_len):
            value = _energy(sequence, init_chain)
            if value < best_energy:
                best_energy = value
                best_chain = init_chain
    
    # Pass 1: Contact-map-guided growth (primary strategy)
    for i in range(num_grow):
        if _time_left() < 35:
            break
        
        # Alternate between constrained and random growth
        if i % 3 == 0:
            bias = 1.0 + rng.random() * 2.5
            chain = _grow_constrained(sequence, rng, contact_pairs, bias=bias)
        elif i % 3 == 1:
            bias = 0.5 + rng.random() * 3.0
            chain = _grow_random(sequence, rng, bias=bias)
        else:
            # Constrained with lower bias (more exploration)
            bias = 0.3 + rng.random() * 1.5
            chain = _grow_constrained(sequence, rng, contact_pairs, bias=bias)
        
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Pass 2: Local search on best
    if best_chain is not None and _time_left() > 38:
        improved = _local_search(sequence, best_chain, max_iterations=20000,
                                  time_limit=ls_time, rng=rng)
        if improved is not None:
            e = _energy(sequence, improved)
            if e < best_energy:
                best_energy = e
                best_chain = improved
        
        # Polish
        if _time_left() > 30:
            polished = _polish(sequence, best_chain, max_iterations=10000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                e = _energy(sequence, polished)
                if e < best_energy:
                    best_energy = e
                    best_chain = polished
    
    # Pass 3: Second round with different seed
    if best_chain is not None and _time_left() > 45:
        rng2 = random.Random(99999 + seq_len)
        improved2 = _local_search(sequence, best_chain, max_iterations=15000,
                                    time_limit=ls_time * 0.8, rng=rng2)
        if improved2 is not None:
            e2 = _energy(sequence, improved2)
            if e2 < best_energy:
                best_energy = e2
                best_chain = improved2
        
        if _time_left() > 35:
            polished2 = _polish(sequence, best_chain, max_iterations=8000,
                                 time_limit=polish_time * 0.7, rng=rng2)
            if polished2 is not None:
                e = _energy(sequence, polished2)
                if e < best_energy:
                    best_energy = e
                    best_chain = polished2
    
    # Pass 4: Additional growth rounds if time permits
    if _time_left() > 50:
        extra = 60
        for i in range(extra):
            if _time_left() < 38:
                break
            bias = 0.5 + rng2.random() * 3.0
            chain = _grow_constrained(sequence, rng2, contact_pairs, bias=bias)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_energy = value
                best_chain = chain
                if _time_left() > 35:
                    polished = _polish(sequence, best_chain, max_iterations=5000,
                                       time_limit=3.0, rng=rng2)
                    if polished is not None:
                        pe = _energy(sequence, polished)
                        if pe < best_energy:
                            best_energy = pe
                            best_chain = polished
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
