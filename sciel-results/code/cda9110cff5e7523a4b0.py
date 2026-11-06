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


def _compute_contact_counts(sequence, chain, occupied):
    """Return the number of H-H contacts."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _slither_delta_contacts(sequence, chain, occupied, new_last):
    """Calculate contact delta (positive = gain) for a slither move."""
    n = len(chain)
    i = n - 1
    p_old = chain[i]
    p_prev = chain[i - 1]

    delta = 0
    if sequence[i] == 'H':
        for q in _neighbours(p_old):
            if q == p_prev:
                continue
            j = occupied.get(q)
            if j is not None and sequence[j] == 'H':
                delta -= 1
        for q in _neighbours(new_last):
            if q == p_prev:
                continue
            j = occupied.get(q)
            if j is not None and sequence[j] == 'H':
                delta += 1
    return delta


def _pivot_delta_contacts(sequence, chain, occupied, pivot_idx, rotated):
    """Calculate contact delta for a pivot move (only counts contacts with fixed part)."""
    delta = 0
    start_idx = pivot_idx + 1
    n = len(chain)

    for i in range(start_idx, n):
        p_old = chain[i]
        p_new = rotated[i - start_idx]

        if sequence[i] == 'H':
            for q in _neighbours(p_old):
                j = occupied.get(q)
                if j is not None and j <= pivot_idx and sequence[j] == 'H':
                    delta -= 1
            for q in _neighbours(p_new):
                j = occupied.get(q)
                if j is not None and j <= pivot_idx and sequence[j] == 'H':
                    delta += 1

    return delta


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


def _contact_map(sequence):
    """Pre-compute a contact map: for each H-H pair, the number of steps between them."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if j - i > 1:
                pairs.append((i, j, j - i))
    return pairs


def _constraint_growth(sequence, dimension, rng, contact_pairs, max_contacts_to_enforce=5):
    """
    Backtracking growth with hard constraints on top contact pairs.
    """
    n = len(sequence)
    if n < 4:
        return _grow(sequence, dimension, rng, bias=1.0)

    sorted_pairs = sorted(contact_pairs, key=lambda x: x[2])
    key_pairs = sorted_pairs[:max_contacts_to_enforce]

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

            constraint_bonus = 0
            for j, k, _ in key_pairs:
                if i == j and k < i and k in occupied:
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


def _local_search(sequence, chain, max_iterations=50000, time_limit=10.0, rng=None):
    """Simulated annealing local search with incremental energy tracking."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_contacts = _compute_contact_counts(sequence, current_chain, current_occupied)
    current_energy = -current_contacts
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 4.0
    cooling = 0.9997
    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break

        move_type = rng.random()

        if move_type < 0.65:
            # Pivot move
            if n < 4:
                temperature *= cooling
                continue
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])

            best_delta = None
            best_rotated = None

            for angle in (90, -90, 180):
                rotated = []
                seen_rotated = set()
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

                delta = _pivot_delta_contacts(sequence, current_chain, current_occupied, pivot_idx, rotated)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_rotated = rotated

            if best_rotated is None:
                temperature *= cooling
                continue

            if best_delta < 0 or rng.random() < math.exp(-best_delta / max(temperature, 0.05)):
                # Accept
                for p in segment:
                    if p in current_occupied:
                        del current_occupied[p]
                for i, p in enumerate(best_rotated):
                    current_occupied[p] = segment_start + i

                current_chain[segment_start:] = best_rotated
                current_contacts += best_delta
                current_energy = -current_contacts

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Slither move
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
            delta = _slither_delta_contacts(sequence, current_chain, current_occupied, new_last)

            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.05)):
                del current_occupied[last]
                current_occupied[new_last] = n - 1
                current_chain[-1] = new_last
                current_contacts += delta
                current_energy = -current_energy * 0 + (-current_contacts)

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, time_limit=5.0, rng=None):
    """High-precision polish pass with low temperature and deterministic end-segment search."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_contacts = _compute_contact_counts(sequence, current_chain, current_occupied)
    current_energy = -current_contacts
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 1.0
    cooling = 0.9998
    n = len(current_chain)

    while time.time() - start_time < time_limit and _time_left() > 15:
        move_type = rng.random()

        if move_type < 0.7:
            # Pivot
            if n < 4:
                temperature *= cooling
                continue
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])

            best_delta = None
            best_rotated = None

            for angle in (90, -90, 180):
                rotated = []
                seen_rotated = set()
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

                delta = _pivot_delta_contacts(sequence, current_chain, current_occupied, pivot_idx, rotated)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_rotated = rotated

            if best_rotated is None:
                temperature *= cooling
                continue

            if best_delta < 0 or (best_delta <= 1 and rng.random() < math.exp(-best_delta / max(temperature, 0.03))):
                for p in segment:
                    if p in current_occupied:
                        del current_occupied[p]
                for i, p in enumerate(best_rotated):
                    current_occupied[p] = segment_start + i

                current_chain[segment_start:] = best_rotated
                current_contacts += best_delta
                current_energy = -current_contacts

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Slither
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
            delta = _slither_delta_contacts(sequence, current_chain, current_occupied, new_last)

            if delta < 0 or (delta <= 1 and rng.random() < math.exp(-delta / max(temperature, 0.03))):
                del current_occupied[last]
                current_occupied[new_last] = n - 1
                current_chain[-1] = new_last
                current_contacts += delta
                current_energy = -current_contacts

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        temperature *= cooling

    # Deterministic end-segment rotation search
    n = len(current_chain)
    for pivot_idx in range(max(1, n - 12), n - 1):
        if _time_left() < 15:
            break
        pivot = current_chain[pivot_idx]
        segment = current_chain[pivot_idx + 1:]
        fixed_set = set(current_chain[:pivot_idx + 1])
        seen_rotated = set()

        best_delta = None
        best_rotated = None

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

            delta = _pivot_delta_contacts(sequence, current_chain, current_occupied, pivot_idx, rotated)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_rotated = rotated

        if best_rotated is not None and best_delta < 0:
            for p in segment:
                if p in current_occupied:
                    del current_occupied[p]
            for i, p in enumerate(best_rotated):
                current_occupied[p] = pivot_idx + 1 + i
            current_chain[pivot_idx + 1:] = best_rotated
            current_contacts += best_delta
            current_energy = -current_contacts
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    if seq_len <= 30:
        num_grow = 250
        local_search_time = 18.0
        polish_time = 5.0
    elif seq_len <= 50:
        num_grow = 200
        local_search_time = 15.0
        polish_time = 4.0
    elif seq_len <= 70:
        num_grow = 150
        local_search_time = 12.0
        polish_time = 3.5
    else:
        num_grow = 100
        local_search_time = 10.0
        polish_time = 3.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pre-compute contact map for constraint-based growth
    contact_pairs = _contact_map(sequence)

    # Pass 1: Greedy Growth (standard) with varied bias
    for i in range(num_grow // 2):
        if _time_left() < 25:
            break
        bias = 0.3 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Constraint-based growth
    for i in range(num_grow // 2):
        if _time_left() < 25:
            break
        chain = _constraint_growth(sequence, dimension, rng, contact_pairs, max_contacts_to_enforce=6)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 3: Local Search on the best chain found
    if best_chain is not None and _time_left() > 35:
        improved = _local_search(sequence, best_chain, max_iterations=100000,
                                 time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Polish pass
        if _time_left() > 25:
            actual_polish_time = min(polish_time, _time_left() - 20)
            if actual_polish_time > 1.0:
                rng3 = random.Random(rng.random() * 1000000)
                polished = _polish(sequence, best_chain, time_limit=actual_polish_time, rng=rng3)
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
