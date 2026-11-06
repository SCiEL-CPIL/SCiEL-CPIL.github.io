"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


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
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _slither_delta(sequence, chain, occupied, new_last):
    """Calculate energy delta for a slither move."""
    n = len(chain)
    i = n - 1
    p_old = chain[i]
    p_prev = chain[i - 1]
    
    delta_contacts = 0
    
    # Loss of contacts from old last point
    if sequence[i] == 'H':
        for q in _neighbours(p_old):
            if q == p_prev:
                continue
            j = occupied.get(q)
            if j is not None and sequence[j] == 'H':
                delta_contacts -= 1
                
    # Gain of contacts from new last point
    if sequence[i] == 'H':
        for q in _neighbours(new_last):
            if q == p_prev:
                continue
            j = occupied.get(q)
            if j is not None and sequence[j] == 'H':
                delta_contacts += 1
                
    return -delta_contacts # Energy is -contacts


def _pivot_delta(sequence, chain, occupied, pivot_idx, new_segment):
    """Calculate energy delta for a pivot move."""
    delta_contacts = 0
    start_idx = pivot_idx + 1
    n = len(chain)
    
    for i in range(start_idx, n):
        p_old = chain[i]
        p_new = new_segment[i - start_idx]
        
        if sequence[i] == 'H':
            # Check neighbors of p_old in F (indices <= pivot_idx)
            for q in _neighbours(p_old):
                j = occupied.get(q)
                if j is not None and j <= pivot_idx and sequence[j] == 'H':
                    delta_contacts -= 1
                    
            # Check neighbors of p_new in F
            for q in _neighbours(p_new):
                j = occupied.get(q)
                if j is not None and j <= pivot_idx and sequence[j] == 'H':
                    delta_contacts += 1
                    
    return -delta_contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')

            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain * math.exp(-0.3 * dist) + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain."""
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
    seen_rotated = set()

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
            if new_p in fixed_set or new_p in seen_rotated:
                valid = False
                break
            seen_rotated.add(new_p)
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


def _contact_map(sequence):
    """Pre-compute a contact map: for each H-H pair, the number of steps between them."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if j - i > 1:  # Not adjacent along chain
                pairs.append((i, j, j - i))
    return pairs


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.999

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot
            n = len(current_chain)
            if n < 4:
                temperature *= cooling
                continue
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])
            seen_rotated = set()
            
            found = False
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
                    if new_p in fixed_set or new_p in seen_rotated:
                        valid = False
                        break
                    seen_rotated.add(new_p)
                    rotated.append(new_p)
                
                if not valid:
                    continue
                
                new_segment = rotated
                new_chain_candidate = current_chain[:segment_start] + rotated
                
                delta_e = _pivot_delta(sequence, current_chain, current_occupied, pivot_idx, rotated)
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temperature, 0.1)):
                    # Accept
                    for p in segment:
                        if p in current_occupied:
                            del current_occupied[p]
                    for i, p in enumerate(rotated):
                        current_occupied[p] = segment_start + i
                        
                    current_chain = new_chain_candidate
                    current_energy += delta_e
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        
                    found = True
                    break
            if found:
                temperature *= cooling
                continue
        else:
            # Slither
            n = len(current_chain)
            if n < 4:
                temperature *= cooling
                continue
            last = current_chain[-1]
            prev = current_chain[-2]
            
            candidates = []
            for site in _neighbours(prev):
                if site in current_occupied:
                    continue
                candidates.append(site)
                
            if not candidates:
                temperature *= cooling
                continue
                
            new_last = rng.choice(candidates)
            delta_e = _slither_delta(sequence, current_chain, current_occupied, new_last)
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temperature, 0.1)):
                # Accept
                del current_occupied[last]
                current_occupied[new_last] = n - 1
                current_chain[-1] = new_last
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        temperature *= cooling

    return best_chain


def _polish(sequence, chain, time_limit=3.0, rng=None):
    """High-precision polish pass with low temperature for fine-tuning."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 1.0
    cooling = 0.9995

    while time.time() - start_time < time_limit and _time_left() > 15:
        r = rng.random()
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot
            n = len(current_chain)
            if n < 4:
                temperature *= cooling
                continue
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])
            seen_rotated = set()
            
            found = False
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
                    if new_p in fixed_set or new_p in seen_rotated:
                        valid = False
                        break
                    seen_rotated.add(new_p)
                    rotated.append(new_p)
                
                if not valid:
                    continue
                
                new_segment = rotated
                new_chain_candidate = current_chain[:segment_start] + rotated
                
                delta_e = _pivot_delta(sequence, current_chain, current_occupied, pivot_idx, rotated)
                
                if delta_e < 0 or (delta_e <= 1 and rng.random() < math.exp(-delta_e / max(temperature, 0.05))):
                    for p in segment:
                        if p in current_occupied:
                            del current_occupied[p]
                    for i, p in enumerate(rotated):
                        current_occupied[p] = segment_start + i
                        
                    current_chain = new_chain_candidate
                    current_energy += delta_e
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        
                    found = True
                    break
            if found:
                temperature *= cooling
                continue
        else:
            # Slither
            n = len(current_chain)
            if n < 4:
                temperature *= cooling
                continue
            last = current_chain[-1]
            prev = current_chain[-2]
            
            candidates = []
            for site in _neighbours(prev):
                if site in current_occupied:
                    continue
                candidates.append(site)
                
            if not candidates:
                temperature *= cooling
                continue
                
            new_last = rng.choice(candidates)
            delta_e = _slither_delta(sequence, current_chain, current_occupied, new_last)
            
            if delta_e < 0 or (delta_e <= 1 and rng.random() < math.exp(-delta_e / max(temperature, 0.05))):
                del current_occupied[last]
                current_occupied[new_last] = n - 1
                current_chain[-1] = new_last
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        temperature *= cooling

    n = len(current_chain)
    for pivot_idx in range(max(1, n - 8), n - 1):
        if _time_left() < 15:
            break
        pivot = current_chain[pivot_idx]
        segment = current_chain[pivot_idx + 1:]
        fixed_set = set(current_chain[:pivot_idx + 1])
        seen_rotated = set()

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
                if new_p in fixed_set or new_p in seen_rotated:
                    valid = False
                    break
                seen_rotated.add(new_p)
                rotated.append(new_p)

            if not valid:
                continue

            new_chain = current_chain[:pivot_idx + 1] + rotated
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

    return best_chain


def _constraint_growth(sequence, dimension, rng, contact_pairs, max_contacts_to_enforce=5):
    """
    Backtracking growth with hard constraints on top contact pairs.
    We try to enforce spatial proximity for the top-k H-H pairs (by closeness in sequence).
    """
    n = len(sequence)
    if n < 4:
        return _grow(sequence, dimension, rng, bias=1.0)

    # Sort contact pairs by sequence distance (closer pairs are easier to satisfy first)
    # Prioritize pairs that are close in sequence
    sorted_pairs = sorted(contact_pairs, key=lambda x: x[2])
    
    # Pick the first few valid pairs as constraints
    key_pairs = sorted_pairs[:max_contacts_to_enforce]
    
    # Build a set of (i, j) pairs that we want to encourage
    constraint_set = set((i, j) for i, j, _ in key_pairs)
    
    # Grow with bias towards satisfying constraints
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            
            # Bonus for being near a constrained partner
            constraint_bonus = 0
            for j, k, _ in key_pairs:
                if i == j and k < i and k in occupied:
                    # We are at position i, and k is already placed
                    # Check if site is close to occupied[k]
                    dist = abs(site[0] - chain[k][0]) + abs(site[1] - chain[k][1])
                    if dist == 1:
                        constraint_bonus += 5.0
                elif i == k and j < i and j in occupied:
                    dist = abs(site[0] - chain[j][0]) + abs(site[1] - chain[j][1])
                    if dist == 1:
                        constraint_bonus += 5.0

            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = 1.5 * gain * math.exp(-0.3 * dist) + 0.1 * free + constraint_bonus + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    if seq_len <= 30:
        num_grow = 200
        local_search_time = 15.0
    elif seq_len <= 50:
        num_grow = 150
        local_search_time = 12.0
    elif seq_len <= 70:
        num_grow = 100
        local_search_time = 10.0
    else:
        num_grow = 80
        local_search_time = 8.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pre-compute contact map for constraint-based growth
    contact_pairs = _contact_map(sequence)

    # Pass 1: Greedy Growth (standard)
    for i in range(num_grow // 2):
        if _time_left() < 20:
            break

        bias = 0.5 + rng.random() * 2.0
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Constraint-based growth
    for i in range(num_grow // 2):
        if _time_left() < 20:
            break

        chain = _constraint_growth(sequence, dimension, rng, contact_pairs, max_contacts_to_enforce=5)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 3: Local Search on the best chain found
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=10000,
                                 time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Polish pass
        if _time_left() > 25:
            polish_time = min(4.0, _time_left() - 20)
            if polish_time > 1.0:
                rng3 = random.Random(rng.random() * 1000000)
                polished = _polish(sequence, best_chain, time_limit=polish_time, rng=rng3)
                if polished is not None:
                    pe = _energy(sequence, polished)
                    if pe < best_energy:
                        best_energy = pe
                        best_chain = polished

    # Final fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
