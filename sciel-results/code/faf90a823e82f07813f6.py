"""HP lattice protein folding: beam growth + simulated-annealing fragment moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 205.0  # wall-clock limit with safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth.
    Keeps the top `beam_width` partial chains based on a heuristic score.
    """
    n = len(sequence)
    if n == 0:
        return []

    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    beam = [(initial_chain, initial_occupied, 0.0)]

    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            current = chain[-1]
            # Generate all valid extensions
            for site in _neighbours(current):
                if site in occupied:
                    continue

                # Calculate immediate gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # Heuristic: favor contacts.
                # We want to maximize gain, so minimize -gain.
                new_score = score - gain * 1.0

                # Add to candidate list
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i

                next_beam.append((new_chain, new_occupied, new_score))

        if not next_beam:
            return None

        # Sort by score (lower is better, i.e., more negative energy potential)
        # Tie-breaker: random to maintain diversity
        next_beam.sort(key=lambda x: (x[2], rng.random()))

        # Keep top beam_width
        beam = next_beam[:beam_width]

        # Early termination if time is low
        if _remaining() < 2.0:
            break

    if not beam:
        return None

    # Pick the best one
    best_chain, _, _ = beam[0]
    return best_chain


def _local_search(sequence, chain, rng, time_budget):
    """
    Local search with simulated annealing using fragment moves.
    Returns improved chain.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    start_time = time.time()
    max_time = time_budget

    T = 5.0  # initial temperature
    T_min = 0.1
    cooling = 0.9995

    steps = 0
    while T > T_min and (time.time() - start_time) < max_time and _remaining() > 2.0:
        # Try a random fragment move
        n = len(current_chain)
        if n < 3:
            break

        max_len = min(10, n - 2)
        frag_len = rng.randint(2, max_len)
        i = rng.randint(1, n - frag_len - 1)
        j = i + frag_len - 1
        axis = rng.randint(0, 1)
        step = rng.choice([-1, 1])

        # Check validity
        occupied_set = set()
        for k in range(n):
            if k < i or k > j:
                occupied_set.add(current_chain[k])

        valid = True
        new_positions = {}
        for k in range(i, j + 1):
            new_pos = list(current_chain[k])
            new_pos[axis] += step
            new_pos = tuple(new_pos)

            if new_pos in occupied_set:
                valid = False
                break
            if new_pos in new_positions.values():
                valid = False
                break
            new_positions[k] = new_pos

        if not valid:
            T *= cooling
            continue

        # Check boundary connectivity
        if i > 0:
            prev_pos = current_chain[i - 1]
            new_i_pos = new_positions[i]
            diff = sum(abs(a - b) for a, b in zip(new_i_pos, prev_pos))
            if diff != 1:
                T *= cooling
                continue

        if j < n - 1:
            next_pos = current_chain[j + 1]
            new_j_pos = new_positions[j]
            diff = sum(abs(a - b) for a, b in zip(new_j_pos, next_pos))
            if diff != 1:
                T *= cooling
                continue

        # Compute delta energy
        where = {p: idx for idx, p in enumerate(current_chain)}
        delta = 0

        for k in range(i, j + 1):
            old_pos = current_chain[k]
            new_pos = new_positions[k]

            if sequence[k] != "H":
                continue

            # Count contacts lost at old position
            for q in _neighbours(old_pos):
                j_idx = where.get(q)
                if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                    if sequence[j_idx] == "H":
                        if not (i <= j_idx <= j):
                            delta += 1

            # Count contacts gained at new position
            for q in _neighbours(new_pos):
                j_idx = where.get(q)
                if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                    if sequence[j_idx] == "H":
                        if not (i <= j_idx <= j):
                            delta -= 1

        delta_e = -delta

        # Accept or reject
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            # Accept move
            current_chain = list(current_chain)
            for k in range(i, j + 1):
                current_chain[k] = new_positions[k]
            current_energy += delta_e

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        T *= cooling
        steps += 1

    return best_chain


def fold(sequence, dimension):
    """Best conformation using beam growth + fragment-move local search with time management."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)

    if n <= 1:
        return [[0]]

    best = None
    best_energy = 0  # worst possible (no contacts)

    # Phase 1: Generate initial conformations via beam growth
    num_growth = min(30, max(10, int(1.5 * n)))
    growth_time_budget = min(5.0, max(1.0, 0.05 * n))

    growth_deadline = time.time() + growth_time_budget
    solutions = []
    for _ in range(num_growth):
        if time.time() > growth_deadline or _remaining() < 5.0:
            break
        chain = _grow_beam(sequence, dimension, rng, beam_width=8)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        solutions.append((value, chain))
        if value < best_energy:
            best_energy = value
            best = chain

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best)
        solutions.append((best_energy, best))

    solutions.sort(key=lambda x: x[0])

    # Phase 2: Local search on top solutions using fragment moves
    top_k = min(5, len(solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = solutions[k]
        ls_time = min(3.0, max(0.5, _remaining() * 0.3 / max(1, top_k - k)))
        improved = _local_search(sequence, chain, rng, ls_time)
        value = _compute_energy(sequence, improved)
        if value < best_energy:
            best_energy = value
            best = improved

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
