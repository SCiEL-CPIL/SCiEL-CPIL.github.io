"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation efficiently."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            j = where.get((x+1, y))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x-1, y))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x, y+1))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x, y-1))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _contact_map(sequence):
    """Pre-compute all potential H-H contact pairs (i,j) with i<j-1."""
    n = len(sequence)
    contacts = []
    for i in range(n):
        if sequence[i] == 'H':
            for j in range(i+2, n):
                if sequence[j] == 'H':
                    contacts.append((i, j))
    return contacts


def _priority_contacts(sequence, top_k=10):
    """Return top-k H-H pairs by sequence distance (longer range = higher priority)."""
    cmap = _contact_map(sequence)
    # Sort by distance (j-i) descending, then by i ascending
    cmap.sort(key=lambda x: (-(x[1]-x[0]), x[0]))
    return cmap[:top_k]


def _polish(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves.
    Uses a more efficient strategy: try random pivots first, then slither."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    stall_count = 0

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        improved = False

        # Try a batch of random pivot moves
        for _ in range(20):
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start >= n:
                continue

            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])

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

                new_chain = current_chain[:segment_start] + rotated
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy

                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Accept first improving move and break inner loop
            if improved:
                break  # Break outer batch loop if we improved

        if not improved:
            # Try slither move
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break

        if not improved:
            stall_count += 1
            if stall_count > 50:
                break
        else:
            stall_count = 0

    return best_chain


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Pre-compute priority contacts for this sequence
    priority_pairs = _priority_contacts(sequence, top_k=15)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            
            # Bonus for satisfying priority long-range contacts
            priority_bonus = 0
            for (pi, pj) in priority_pairs:
                if pj == i and pi < i - 1:
                    # Check if placing site adjacent to chain[pi] would create contact
                    pi_x, pi_y = chain[pi]
                    if (site[0] in (pi_x-1, pi_x+1, pi_x) and 
                        site[1] in (pi_y-1, pi_y+1, pi_y) and
                        abs(site[0]-pi_x) + abs(site[1]-pi_y) == 1):
                        priority_bonus += 2.0
            
            # Heuristic: Prefer sites that are "inside" the current bounding box or close to centroid
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Strong bias towards gaining contacts, secondary bias towards compactness (lower dist)
            score = bias * gain + priority_bonus - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
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
            else:  # 180
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
            
        new_chain = chain[:segment_start] + rotated
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _local_search(sequence, chain, max_iterations=10000, time_limit=8.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.998  # Slower cooling to explore more

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move: 60% Pivot, 20% Slither, 20% Crankshaft (simplified pivot)
        r = rng.random()
        if r < 0.6:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.8:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move_fast(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _spiral_fold(sequence):
    """Create a compact spiral conformation as a deterministic starting point."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (1, 0)]
    
    # Build a spiral that stays compact
    chain = [(0, 0)]
    # Directions: right, up, left, down, repeat but expanding
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_per_dir = 1
    
    while len(chain) < n:
        dx, dy = directions[dir_idx]
        # Take steps_per_dir steps in this direction
        for _ in range(steps_per_dir):
            if len(chain) >= n:
                break
            last_x, last_y = chain[-1]
            new_x = last_x + dx
            new_y = last_y + dy
            chain.append((new_x, new_y))
        
        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 2:  # After going left, increase steps
            steps_per_dir += 1
    
    return chain[:n]


def _snake_fold(sequence):
    """Create a snake-like conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    # Snake pattern: go right, down, left, down, right, down...
    width = max(1, int(math.sqrt(n)) + 1)
    
    x, y = 0, 0
    chain.append((x, y))
    
    for i in range(1, n):
        if i % (2 * width) < width:
            # Going right
            x = (i % width) if (i // width) % 2 == 0 else width - 1 - (i % width)
            y = i // width
        else:
            # Going left
            x = width - 1 - ((i - width) % width) if (i // width) % 2 == 1 else (i - width) % width
            y = i // width
        
        # Simplified: just use a proper snake
        row = i // width
        col = i % width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        
        chain.append((x, y))
    
    return chain


def _backtrack_search(sequence, max_depth=50, time_limit=8.0, rng=None):
    """Constraint-satisfaction backtracking search targeting priority H-H contacts."""
    if rng is None:
        rng = random.Random(42)
    
    n = len(sequence)
    if n < 4:
        return None
    
    # Pre-compute priority contacts
    priority_pairs = _priority_contacts(sequence, top_k=10)
    
    # Start with a random valid prefix
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Try multiple random prefixes
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 20:
            break
        
        # Reset for new attempt
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # Build a random valid prefix of length min(10, n//2)
        prefix_len = min(10, n // 2)
        valid_prefix = True
        for i in range(1, prefix_len):
            options = []
            for site in _neighbours(chain[-1]):
                if site not in occupied:
                    options.append(site)
            if not options:
                valid_prefix = False
                break
            site = options[rng.randint(0, len(options)-1)]
            occupied[site] = i
            chain.append(site)
        
        if not valid_prefix:
            continue
        
        # Now try to complete the chain with backtracking, prioritizing contacts
        def _backtrack(i):
            nonlocal best_chain, best_energy
            if time.time() - start_time > time_limit:
                return False
            if _time_left() < 15:
                return False
            
            if i == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
                return True
            
            # Get candidate sites
            candidates = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                
                # Score this site
                score = 0
                if sequence[i] == 'H':
                    # Count immediate H-H contacts
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            score += 3.0
                    
                    # Bonus for priority long-range contacts
                    for (pi, pj) in priority_pairs:
                        if pj == i and pi < i - 1:
                            pi_x, pi_y = chain[pi]
                            if abs(site[0]-pi_x) + abs(site[1]-pi_y) == 1:
                                score += 5.0
                
                # Compactness bonus
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = (site[0]-cx)**2 + (site[1]-cy)**2
                score -= 0.1 * dist
                
                candidates.append((score, site))
            
            if not candidates:
                return False
            
            # Sort by score descending, with some randomness
            candidates.sort(key=lambda x: -x[0])
            
            # Try top candidates with some randomization
            trial_count = min(5, len(candidates))
            for k in range(trial_count):
                if time.time() - start_time > time_limit:
                    return False
                if _time_left() < 15:
                    return False
                
                # Add randomness for diversity
                if k == 0:
                    idx = 0
                else:
                    idx = rng.randint(0, len(candidates)-1)
                
                site = candidates[idx][1]
                chain.append(site)
                occupied[site] = i
                
                if _backtrack(i + 1):
                    # Found a solution, but keep searching for better ones
                    pass
                
                chain.pop()
                del occupied[site]
            
            return False
        
        _backtrack(prefix_len)
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 500
        local_search_time = 8.0
        polish_time = 4.0
        backtrack_time = 5.0
    elif seq_len <= 50:
        num_grow = 350
        local_search_time = 7.0
        polish_time = 3.5
        backtrack_time = 4.0
    elif seq_len <= 70:
        num_grow = 250
        local_search_time = 6.0
        polish_time = 3.0
        backtrack_time = 3.0
    else:
        num_grow = 180
        local_search_time = 5.0
        polish_time = 2.5
        backtrack_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 0: Try deterministic structures first
    for init_chain in [_spiral_fold(sequence), _snake_fold(sequence)]:
        if init_chain is not None and len(init_chain) == seq_len:
            # Verify self-avoidance
            if len(set(init_chain)) == seq_len:
                # Verify unit steps
                valid = True
                for i in range(1, seq_len):
                    dx = abs(init_chain[i][0] - init_chain[i-1][0])
                    dy = abs(init_chain[i][1] - init_chain[i-1][1])
                    if dx + dy != 1:
                        valid = False
                        break
                if valid:
                    value = _energy(sequence, init_chain)
                    if value < best_energy:
                        best_energy = value
                        best_chain = init_chain

    # Pass 1: Backtracking search with contact map guidance
    if _time_left() > 35:
        bt_chain = _backtrack_search(sequence, time_limit=backtrack_time, rng=rng)
        if bt_chain is not None and len(bt_chain) == seq_len:
            bt_energy = _energy(sequence, bt_chain)
            if bt_energy < best_energy:
                best_energy = bt_energy
                best_chain = bt_chain

    # Pass 2: Greedy Growth
    # We use a range of biases to explore different compactness levels
    for i in range(num_grow):
        if _time_left() < 25:
            break

        # Bias between 0.5 and 3.0
        bias = 0.5 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 3: Local Search on the best chain found
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=12000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Deterministic polish
        if _time_left() > 25:
            polished = _polish(sequence, best_chain, max_iterations=6000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 5: Additional local search with different seed if time permits
        if _time_left() > 35:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=6000,
                                       time_limit=local_search_time * 0.7, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 25:
                final_polish = _polish(sequence, best_chain, max_iterations=4000,
                                       time_limit=polish_time * 0.6, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 6: Another round of growth + search if time permits
        if _time_left() > 40:
            extra_grow = 40
            for i in range(extra_grow):
                if _time_left() < 30:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick polish
                    if _time_left() > 25:
                        polished = _polish(sequence, best_chain, max_iterations=2500,
                                           time_limit=1.5, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
