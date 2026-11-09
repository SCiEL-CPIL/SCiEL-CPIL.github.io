"""HP lattice protein folding: constraint-satisfaction backtracking with lookahead + local refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute total energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, (x, y) in enumerate(chain):
        if (x, y) in seen:
            return False
        seen.add((x, y))
        if i > 0:
            px, py = chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                return False
    return True


def _backtrack_fold(sequence, time_limit, rng):
    """
    Backtracking search with lookahead heuristic.
    Places residues one by one, prioritizing positions that maximize potential H-H contacts.
    Uses constraint propagation to prune infeasible branches early.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n

    deadline = time.time() + time_limit
    best_chain = None
    best_energy = float('inf')

    # Precompute H indices
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    h_count = len(h_indices)

    # Start with the first residue at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    def _lookahead_score(candidate_pos, current_idx, chain_so_far, occupied_so_far):
        """
        Score a candidate position for the next residue.
        Higher score = more promising.
        """
        if current_idx >= n - 1:
            return 0

        score = 0
        x, y = candidate_pos

        # If current residue is H, count potential H neighbors already placed
        if sequence[current_idx] == 'H':
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = occupied_so_far.get(q)
                if j is not None and j < current_idx - 1:
                    if sequence[j] == 'H':
                        score += 10  # Strong preference for immediate H-H contact

        # If next residue is H, prefer positions adjacent to existing H residues
        if current_idx + 1 < n and sequence[current_idx + 1] == 'H':
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = occupied_so_far.get(q)
                if j is not None and sequence[j] == 'H':
                    score += 5

        # Prefer positions with more free neighbors (less trapped)
        free = 0
        for dx, dy in _DELTAS:
            if (x + dx, y + dy) not in occupied_so_far:
                free += 1
        score += free * 0.5

        # Small random noise to avoid determinism
        score += rng.random() * 0.1

        return score

    def _search(depth):
        nonlocal best_chain, best_energy

        if time.time() > deadline:
            return

        if depth == n:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            return

        # Generate candidate positions for residue at 'depth'
        candidates = []
        last_x, last_y = chain[-1]
        for dx, dy in _DELTAS:
            cand = (last_x + dx, last_y + dy)
            if cand not in occupied:
                candidates.append(cand)

        if not candidates:
            return

        # Score and sort candidates
        scored = []
        for cand in candidates:
            s = _lookahead_score(cand, depth, chain, occupied)
            scored.append((-s, cand))  # Min-heap: most negative = highest score
        scored.sort()

        for _, cand in scored:
            if time.time() > deadline:
                return

            # Try placing residue at cand
            chain.append(cand)
            occupied[cand] = depth

            # Recurse
            _search(depth + 1)

            # Backtrack
            chain.pop()
            del occupied[cand]

    # Run backtracking search
    _search(1)

    if best_chain is None:
        # Fallback: simple growth
        best_chain = _greedy_grow(sequence, rng)

    return best_chain


def _greedy_grow(sequence, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
    """Greedy growth with bias toward hydrophobic contacts."""
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        options = []
        for site in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = chain[-1]
            cand = (x + site[0], y + site[1])
            if cand in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                cx, cy = cand
                for dx, dy in _DELTAS:
                    q = (cx + dx, cy + dy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            cx, cy = cand
            for dx, dy in _DELTAS:
                if (cx + dx, cy + dy) not in occupied:
                    free += 1
            score = hh_weight * gain + free_weight * free + noise * rng.random()
            options.append((score, cand))
        if not options:
            return None
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _local_refine(sequence, chain, time_limit, rng):
    """Local refinement using 2-opt reversals and simulated annealing."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    T = 1.5
    cooling = 0.99998
    min_t = 0.01

    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 3000

    h_indices = [i for i in range(n) if sequence[i] == 'H']

    def _try_reversal(i, j):
        """Try reversing segment [i, j]. Return new chain or None if invalid."""
        if i >= j:
            return None
        # Check boundary connectivity
        if i > 0:
            px, py = chain[i - 1]
            ex, ey = chain[j]
            if abs(px - ex) + abs(py - ey) != 1:
                return None
        if j < n - 1:
            sx, sy = chain[i]
            nx, ny = chain[j + 1]
            if abs(sx - nx) + abs(sy - ny) != 1:
                return None
        # Check self-avoidance of the reversed segment
        segment = chain[i:j+1]
        for pos in segment:
            if pos in chain[:i] or pos in chain[j+1:]:
                return None
        new_chain = chain[:i] + segment[::-1] + chain[j+1:]
        return new_chain

    while time.time() < deadline:
        if _remaining_time() < 2.0:
            break

        # Generate a move
        if h_indices and rng.random() < 0.4:
            hi = rng.choice(h_indices)
            span = rng.randint(2, min(15, n - 1))
            i = max(0, hi - span // 2)
            j = min(n - 1, i + span)
            if j <= i:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
        else:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)

        new_chain = _try_reversal(i, j)
        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta <= 0 or (T > min_t and rng.random() < math.exp(-delta / T)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            break

        T *= cooling
        if T < min_t:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    time_limit = max(0.5, _remaining_time() - 5.0)

    # Phase 1: Backtracking search with lookahead
    bt_deadline = time.time() + min(30.0, time_limit * 0.4)
    bt_chain = _backtrack_fold(sequence, bt_deadline - time.time(), rng)
    bt_energy = _energy(sequence, bt_chain)

    # Phase 2: Generate diverse initial chains via greedy growth
    initial_chains = [bt_chain]
    configs = [
        (2.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.5, 0.2, 0.3),
        (1.0, 0.3, 1.0),
        (3.0, 0.0, 0.2),
    ]
    growth_deadline = time.time() + min(5.0, time_limit * 0.15)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(2):
            if time.time() > growth_deadline:
                break
            chain = _greedy_grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    # Score and select top candidates
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(5, len(scored))]]

    # Phase 3: Local refinement on each candidate
    best_chain = None
    best_energy = float('inf')
    total_time = _remaining_time() - 3.0
    time_per = total_time / len(candidates) if candidates else 0

    for chain in candidates:
        if _remaining_time() < 2.0:
            break
        alloc = min(time_per, _remaining_time() - 1.0)
        if alloc < 0.3:
            continue
        refined_chain, refined_energy = _local_refine(sequence, chain, alloc, rng)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    # Validate
    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
