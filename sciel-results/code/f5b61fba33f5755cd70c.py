"""HP lattice folding: beam search growth + multi-move local search."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.time()
_DEADLINE = 195.0


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, rng):
    """
    Beam search growth: maintain beam_width partial conformations,
    expanding each by one residue and keeping the best by score.
    Returns list of (energy, chain) for complete conformations.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, [(0, 0)])]

    # Each beam entry: (score, chain, occupied_set)
    start = (0, 0)
    beam = [(-1, [start], {start})]  # score = contacts so far (we minimize -contacts)

    for i in range(1, n):
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                # Compute new contacts gained
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = new_chain.index(q) if q in new_chain else -1
                            # Need to check if the occupant is H and not adjacent
                            # occupied maps position to... we need index. Let's use a dict.
                            pass
                # Simpler: just compute energy of the partial chain up to i
                # But that's expensive. Let's track contacts incrementally.
                # For now, use a proxy: count H-H neighbors among already placed
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find which residue is at q
                            # We need position->index mapping; let's rebuild for beam search
                            pass

        # The above is getting complex. Let's use a cleaner approach:
        # Store (energy_so_far, chain, pos_to_idx) in beam
        new_beam = []
        for entry in beam:
            score, chain, pos_to_idx, occupied = entry
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                # Compute delta energy
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = new_pos_to_idx.get(q, -1)
                        if j >= 0 and j < i - 1 and sequence[j] == "H":
                            delta += 1
                new_score = score - delta  # energy = -contacts, so more contacts = lower energy
                new_beam.append((new_score, new_chain, new_pos_to_idx, new_occupied))

        if not new_beam:
            break
        # Keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return []

    results = []
    for score, chain, pos_to_idx, occupied in beam:
        energy = _compute_energy(sequence, chain)
        results.append((energy, chain))

    return results


def _beam_search_v2(sequence, dimension, beam_width, rng):
    """Cleaner beam search with incremental energy tracking."""
    n = len(sequence)
    if n <= 1:
        return [(0, [(0, 0)])]

    # Beam entry: (energy, chain, pos_to_idx)
    start = (0, 0)
    beam = [(0, [start], {start: 0})]

    for i in range(1, n):
        new_beam = []
        for energy, chain, pos_to_idx in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in pos_to_idx:
                    continue
                new_chain = chain + [site]
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                # Compute delta energy: new contacts formed by residue i at site
                delta_contacts = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = new_pos_to_idx.get(q, -1)
                        if 0 <= j <= i - 2 and sequence[j] == "H":
                            delta_contacts += 1
                new_energy = energy - delta_contacts
                new_beam.append((new_energy, new_chain, new_pos_to_idx))

        if not new_beam:
            return []

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return []

    results = []
    for energy, chain, pos_to_idx in beam:
        # Recompute exact energy to be safe
        exact_energy = _compute_energy(sequence, chain)
        results.append((exact_energy, chain))

    return results


def _local_search_v2(sequence, chain, rng, time_budget):
    """
    Local search with kink moves, loop moves, and simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    start_time = time.time()
    T = 3.0
    T_min = 0.05
    cooling = 0.9995

    while T > T_min and (time.time() - start_time) < time_budget and _remaining() > 2.0:
        move_type = rng.random()
        if move_type < 0.6:
            # Kink move: move one residue
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            occupied = set(current_chain)
            candidates = []
            for q in _neighbours(old_pos):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q not in occupied:
                    candidates.append(q)
            if not candidates:
                T *= cooling
                continue
            new_pos = rng.choice(candidates)
            # Verify self-avoidance after move
            test_chain = current_chain[:]
            test_chain[i] = new_pos
            if len(set(test_chain)) != n:
                T *= cooling
                continue
            # Compute energy delta
            pos_to_idx = {p: idx for idx, p in enumerate(current_chain)}
            delta = 0
            if sequence[i] == "H":
                for q in _neighbours(old_pos):
                    j = pos_to_idx.get(q, -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == "H":
                        delta += 1  # losing contact
                for q in _neighbours(new_pos):
                    j = pos_to_idx.get(q, -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == "H":
                        delta -= 1  # gaining contact
            else:
                for q in _neighbours(old_pos):
                    j = pos_to_idx.get(q, -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == "H":
                        delta += 1
                for q in _neighbours(new_pos):
                    j = pos_to_idx.get(q, -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == "H":
                        delta -= 1
            delta_e = -delta
            if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                current_chain[i] = new_pos
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        elif move_type < 0.85:
            # Reversal move: reverse a segment
            if n < 4:
                T *= cooling
                continue
            i = rng.randint(1, n - 3)
            length = rng.randint(2, min(n - i - 1, 8))
            j = i + length - 1
            if j >= n - 1:
                T *= cooling
                continue
            # Try reversing chain[i:j+1]
            segment = current_chain[i:j + 1]
            reversed_segment = segment[::-1]
            # Check if reversal is valid (no self-avoidance violations)
            test_chain = current_chain[:]
            test_chain[i:j + 1] = reversed_segment
            if len(set(test_chain)) != n:
                T *= cooling
                continue
            # Check unit steps at boundaries
            if i > 0:
                p1 = test_chain[i - 1]
                p2 = test_chain[i]
                if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                    T *= cooling
                    continue
            if j < n - 1:
                p1 = test_chain[j]
                p2 = test_chain[j + 1]
                if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                    T *= cooling
                    continue
            new_energy = _compute_energy(sequence, test_chain)
            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                current_chain = test_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Fragment re-insertion: pick a short fragment, remove it, try re-inserting elsewhere
            if n < 6:
                T *= cooling
                continue
            frag_len = rng.randint(2, min(5, n // 3))
            i = rng.randint(1, n - frag_len - 1)
            fragment = current_chain[i:i + frag_len]
            # Remove fragment
            remaining = current_chain[:i] + current_chain[i + frag_len:]
            if len(set(remaining)) != len(remaining):
                T *= cooling
                continue
            # Try inserting fragment at various positions
            best_insert = None
            best_insert_energy = current_energy
            for pos in range(1, len(remaining) - frag_len + 1):
                # Check if insertion creates valid chain
                test_chain = remaining[:pos] + fragment + remaining[pos:]
                if len(set(test_chain)) != n:
                    continue
                # Check unit steps
                valid = True
                for k in range(n - 1):
                    if sum(abs(a - b) for a, b in zip(test_chain[k], test_chain[k + 1])) != 1:
                        valid = False
                        break
                if not valid:
                    continue
                e = _compute_energy(sequence, test_chain)
                if e < best_insert_energy:
                    best_insert_energy = e
                    best_insert = test_chain
            if best_insert is not None:
                delta_e = best_insert_energy - current_energy
                if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                    current_chain = best_insert
                    current_energy = best_insert_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        T *= cooling

    return best_chain


def fold(sequence, dimension):
    """Best conformation using beam search + multi-move local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)

    if n <= 1:
        return [[0]]

    best = None
    best_energy = 0

    # Phase 1: Beam search with varying widths
    beam_configs = [
        (5, 1),    # narrow beam, 1 run
        (10, 1),   # medium beam
        (20, 1),   # wide beam
        (10, 2),   # medium beam, 2 runs
    ]

    solutions = []
    for beam_width, num_runs in beam_configs:
        if _remaining() < 5.0:
            break
        for run in range(num_runs):
            if _remaining() < 3.0:
                break
            results = _beam_search_v2(sequence, dimension, beam_width, rng)
            for energy, chain in results:
                solutions.append((energy, chain))
                if energy < best_energy:
                    best_energy = energy
                    best = chain

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best)

    # Deduplicate and sort solutions
    seen = set()
    unique_solutions = []
    for energy, chain in solutions:
        key = tuple(chain)
        if key not in seen:
            seen.add(key)
            unique_solutions.append((energy, chain))
    unique_solutions.sort(key=lambda x: x[0])

    # Phase 2: Local search on top solutions
    top_k = min(6, len(unique_solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = unique_solutions[k]
        ls_time = min(4.0, max(0.5, _remaining() * 0.25 / max(1, top_k - k)))
        improved = _local_search_v2(sequence, chain, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved

    # Final local search on best
    if _remaining() > 1.0:
        ls_time = min(3.0, _remaining() * 0.4)
        improved = _local_search_v2(sequence, best, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
