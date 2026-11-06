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
    """Return the 4 neighbours of a 2D lattice point as a tuple."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

_NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in _NEIGHBOUR_OFFSETS:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _energy_from_occupied(sequence, chain, occupied):
    """Compute energy using pre-built occupied dict for speed."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in _NEIGHBOUR_OFFSETS:
            j = occupied.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _build_contact_priorities(sequence):
    """Build a priority list of H-H contact pairs based on feasibility.
    Returns a set of (i, j) pairs where i < j, both H, and |i-j| >= 2.
    Weighted by how 'tight' the constraint is (shorter gaps are more valuable).
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    n = len(sequence)
    priorities = set()
    for idx_a, i in enumerate(h_indices):
        for j in h_indices[idx_a + 1:]:
            gap = j - i
            if gap >= 2 and gap <= n - 2:
                # Shorter gaps are more constrained and thus more valuable to satisfy
                priorities.add((i, j))
    return priorities


def _compute_contact_gains(sequence, chain, pos_idx, site, occupied):
    """Compute how many new H-H contacts are formed by placing residue pos_idx at site."""
    if sequence[pos_idx] != 'H':
        return 0
    gains = 0
    x, y = site
    for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        j = occupied.get(q)
        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
            gains += 1
    return gains


def _grow(sequence, dimension, rng, bias=1.0, contact_priorities=None):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    Uses contact_priorities to bias toward satisfying high-priority contacts.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Track centroid incrementally
    cx, cy = 0.0, 0.0
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        # Update centroid incrementally
        cx = (cx * i + last[0]) / (i + 1)
        cy = (cy * i + last[1]) / (i + 1)
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = _compute_contact_gains(sequence, chain, i, site, occupied)
            
            # Bonus for satisfying high-priority contacts
            priority_bonus = 0
            if contact_priorities and sequence[i] == 'H':
                for (a, b) in contact_priorities:
                    if (a == i and b > i) or (b == i and a < i):
                        other = b if a == i else a
                        if other < i:
                            ox, oy = chain[other]
                            sx, sy = site
                            if abs(ox - sx) + abs(oy - sy) == 1:
                                priority_bonus += 2.0
            
            # Compactness: distance from centroid
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = bias * gain + priority_bonus - 0.4 * dist + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _apply_pivot(chain, pivot_idx, angle):
    """Apply a pivot move and return the new chain if valid, else None."""
    n = len(chain)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
    
    fixed_set = set(chain[:pivot_idx + 1])
    segment = chain[segment_start:]
    
    rotated = []
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
            return None
        rotated.append(new_p)
    
    return chain[:segment_start] + rotated


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
    n = len(chain)
    if n < 4:
        return None

    pivot_idx = rng.randint(1, n - 2)
    angle = rng.choice((90, -90, 180))
    return _apply_pivot(chain, pivot_idx, angle)


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    candidates = [site for site in _neighbours(prev) if site not in occupied]
    if not candidates:
        return None
    site = rng.choice(candidates)
    return chain[:-1] + [site]


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Precompute H indices for faster energy computation
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    h_set = set(h_indices)

    start_time = time.time()
    temperature = 4.0
    cooling = 0.9995  # Slower cooling
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.65:
            new_chain = _pivot_move_fast(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)

        if new_chain is None:
            continue

        # Build occupied dict once for energy computation
        occupied = {p: i for i, p in enumerate(new_chain)}
        contacts = 0
        for i in h_indices:
            x, y = new_chain[i]
            for dx, dy in _NEIGHBOUR_OFFSETS:
                j = occupied.get((x + dx, y + dy))
                if j is not None and j > i + 1 and j in h_set:
                    contacts += 1
        new_energy = -contacts
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, max_iterations=2000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Precompute H indices
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    h_set = set(h_indices)

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        best_move = None
        best_delta = 0

        # Try ALL pivot positions for thorough local search
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain is None:
                    continue
                occupied = {p: i for i, p in enumerate(new_chain)}
                contacts = 0
                for i in h_indices:
                    x, y = new_chain[i]
                    for dx, dy in _NEIGHBOUR_OFFSETS:
                        j = occupied.get((x + dx, y + dy))
                        if j is not None and j > i + 1 and j in h_set:
                            contacts += 1
                new_energy = -contacts
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither
        slither = _slither_move(current_chain, rng)
        if slither is not None:
            occupied = {p: i for i, p in enumerate(slither)}
            contacts = 0
            for i in h_indices:
                x, y = slither[i]
                for dx, dy in _NEIGHBOUR_OFFSETS:
                    j = occupied.get((x + dx, y + dy))
                    if j is not None and j > i + 1 and j in h_set:
                        contacts += 1
            new_energy = -contacts
            delta = new_energy - current_energy
            if delta < best_delta:
                best_delta = delta
                best_move = slither

        if best_move is None:
            break

        current_chain = best_move
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Build contact priorities
    contact_priorities = _build_contact_priorities(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 350
        local_search_time = 8.0
        polish_time = 4.0
    elif seq_len <= 50:
        num_grow = 220
        local_search_time = 6.5
        polish_time = 3.5
    elif seq_len <= 70:
        num_grow = 160
        local_search_time = 5.5
        polish_time = 3.0
    else:
        num_grow = 110
        local_search_time = 4.5
        polish_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth with contact priorities
    for i in range(num_grow):
        if _time_left() < 20:
            break

        bias = 0.5 + rng.random() * 2.0
        chain = _grow(sequence, dimension, rng, bias=bias, contact_priorities=contact_priorities)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=15000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish (now tries all pivots)
        if _time_left() > 20:
            polished = _polish(sequence, best_chain, max_iterations=8000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=8000,
                                       time_limit=local_search_time * 0.6, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 20:
                final_polish = _polish(sequence, best_chain, max_iterations=5000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
