"""HP lattice protein folding: progressive segment relocation + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _segment_relocation(chain, sequence, rng, seg_len, max_attempts=50):
    """
    Try to relocate a segment of length seg_len. Cut out chain[a:a+seg_len],
    then attempt to regrow it connecting chain[a-1] to chain[a+seg_len].
    Multiple random attempts to increase success rate.
    Returns new_chain or None.
    """
    n = len(chain)
    if n < seg_len + 2:
        return None

    a = rng.randint(1, n - seg_len - 1)
    b = a + seg_len  # b is the index after the segment

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    # Check Manhattan distance feasibility
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > seg_len or (seg_len - manhattan) % 2 != 0:
        return None

    rest_occupied = set(chain[:a] + chain[b:])

    # Try multiple random DFS attempts
    for attempt in range(max_attempts):
        def _dfs_rand(current, remaining, path, occupied, depth=0):
            if remaining == 0:
                if current == right_neighbor:
                    return list(path)
                return None

            x, y = current
            neighbors = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                dist_to_end = abs(right_neighbor[0] - site[0]) + abs(right_neighbor[1] - site[1])
                if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                    continue
                neighbors.append((dist_to_end, site))

            if not neighbors:
                return None

            # Shuffle for randomness, but sort by distance for heuristic
            rng.shuffle(neighbors)
            neighbors.sort(key=lambda x: x[0])

            for _, site in neighbors:
                path.append(site)
                occupied.add(site)
                result = _dfs_rand(site, remaining - 1, path, occupied, depth + 1)
                if result is not None:
                    return result
                path.pop()
                occupied.discard(site)
            return None

        path = _dfs_rand(left_neighbor, seg_len, [left_neighbor], {left_neighbor} | rest_occupied)
        if path is not None and path[-1] == right_neighbor:
            new_frag = path[1:]
            if len(new_frag) == seg_len:
                new_chain = chain[:a] + new_frag + chain[b:]
                if _is_valid_chain(new_chain):
                    return new_chain
    return None


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Kink move."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    occupied = set(chain)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in occupied and site != chain[i]:
            continue
        if i > 0:
            prev = chain[i - 1]
            if abs(site[0] - prev[0]) + abs(site[1] - prev[1]) != 1:
                continue
        if i < n - 1:
            nxt = chain[i + 1]
            if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) != 1:
                continue
        options.append(site)
    if not options:
        return None
    site = rng.choice(options)
    new_chain = list(chain)
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _pivot_move(chain, rng):
    """Pivot move."""
    n = len(chain)
    if n < 4:
        return None
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    rotations = []
    prefix_set = set(chain[:pivot+1])
    for rot in range(1, 4):
        rotated = []
        valid = True
        for j, (x, y) in enumerate(suffix):
            rx, ry = x - pivot_pos[0], y - pivot_pos[1]
            if rot == 1:
                new_x, new_y = -ry, rx
            elif rot == 2:
                new_x, new_y = -rx, -ry
            else:
                new_x, new_y = ry, -rx
            pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
            if pos in prefix_set:
                valid = False
                break
            if pos in rotated:
                valid = False
                break
            if j > 0:
                prev = rotated[-1]
                if abs(pos[0] - prev[0]) + abs(pos[1] - prev[1]) != 1:
                    valid = False
                    break
            rotated.append(pos)
        if valid and len(rotated) == len(suffix):
            new_chain = chain[:pivot + 1] + rotated
            if len(set(new_chain)) == n:
                rotations.append(new_chain)
    if not rotations:
        return None
    return rng.choice(rotations)


def _try_local_search(sequence, chain, rng, max_iters=2000):
    """
    Local search with simulated annealing and progressive segment relocation.
    Phase 1: Larger segments for coarse optimization.
    Phase 2: Smaller segments for fine-tuning (polish phase).
    """
    n = len(chain)
    current_energy = _energy_fast(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 5.0
    cooling = 0.9995

    # Progressive segment lengths: start large, end small
    half_iters = max_iters // 2

    for it in range(max_iters):
        if _remaining() < 5.0:
            break

        # Determine segment length based on progress
        progress = it / max_iters
        if progress < 0.3:
            max_seg = min(8, n - 3)
        elif progress < 0.6:
            max_seg = min(5, n - 3)
        else:
            max_seg = min(3, n - 3)

        # Choose a move
        r = rng.random()
        if r < 0.5:
            # Segment relocation (primary move)
            seg_len = rng.randint(2, max_seg)
            new_chain = _segment_relocation(chain, sequence, rng, seg_len, max_attempts=30)
        elif r < 0.65:
            # Reversal
            new_chain = _reversal_move(chain, rng)
        elif r < 0.8:
            # Kink
            new_chain = _kink_move(chain, rng)
        else:
            # Pivot
            new_chain = _pivot_move(chain, rng)

        if new_chain is None:
            continue

        # Verify validity
        if len(set(new_chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue

        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - current_energy

        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)

        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using progressive segment relocation."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0

    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)

    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 40
    elif n <= 50:
        num_restarts = 25
    elif n <= 70:
        num_restarts = 18
    else:
        num_restarts = 12

    iters_per_restart = max(1000, int(800000 / n))

    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None or len(chain) != n:
            continue

        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue

        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)

        if energy < best_energy:
            best_energy = energy
            best_chain = chain

        if time.time() - start_time > time_budget:
            break

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
