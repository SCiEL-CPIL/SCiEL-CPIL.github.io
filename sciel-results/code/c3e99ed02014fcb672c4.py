"""HP lattice protein folding on the cubic lattice.
Strategy: Beam Search (fixed width 50) + local refinement with kink moves.
"""
# EVOLVE-BLOCK-START
import time
import random

_TOTAL_DEADLINE = 325.0  # seconds wall-clock (evaluator kills at 330s)
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s safety margin


def _neighbours(point):
    """Yield 6 neighbours of a 3D point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _beam_search(sequence, dimension, beam_width=50):
    """
    Beam search for HP lattice protein folding.
    Returns the best chain found (list of tuples).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start_pos = (0, 0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_pos_to_idx = {start_pos: 0}

    # Beam entries: (energy, chain, occupied, pos_to_idx)
    beam = [(0, initial_chain, initial_occupied, initial_pos_to_idx)]

    for step in range(1, n):
        if _time_remaining() < 0.5:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])

        new_beam = []
        current_residue = sequence[step]

        for energy, chain, occupied, pos_to_idx in beam:
            last_pos = chain[-1]
            prev_pos = chain[-2] if step > 1 else None

            candidates = [
                (last_pos[0] + 1, last_pos[1], last_pos[2]),
                (last_pos[0] - 1, last_pos[1], last_pos[2]),
                (last_pos[0], last_pos[1] + 1, last_pos[2]),
                (last_pos[0], last_pos[1] - 1, last_pos[2]),
                (last_pos[0], last_pos[1], last_pos[2] + 1),
                (last_pos[0], last_pos[1], last_pos[2] - 1),
            ]

            for next_pos in candidates:
                if next_pos in occupied:
                    continue

                delta_e = 0
                if current_residue == 'H':
                    nbs = [
                        (next_pos[0] + 1, next_pos[1], next_pos[2]),
                        (next_pos[0] - 1, next_pos[1], next_pos[2]),
                        (next_pos[0], next_pos[1] + 1, next_pos[2]),
                        (next_pos[0], next_pos[1] - 1, next_pos[2]),
                        (next_pos[0], next_pos[1], next_pos[2] + 1),
                        (next_pos[0], next_pos[1], next_pos[2] - 1),
                    ]
                    for nb in nbs:
                        if nb == prev_pos:
                            continue
                        idx = pos_to_idx.get(nb)
                        if idx is not None:
                            if sequence[idx] == 'H':
                                delta_e += 1

                new_energy = energy - delta_e
                new_chain = chain + [next_pos]
                new_occupied = occupied | {next_pos}
                new_pos_to_idx = pos_to_idx.copy()
                new_pos_to_idx[next_pos] = step

                new_beam.append((new_energy, new_chain, new_occupied, new_pos_to_idx))

        if not new_beam:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

        if step % 10 == 0 and _time_remaining() < 5.0:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])

    best_state = min(beam, key=lambda x: x[0])
    return list(best_state[1])


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_refine(sequence, chain, time_budget=2.0):
    """
    Local refinement using kink moves (move a single residue to an adjacent free site).
    Accepts improving moves; uses simple simulated annealing for occasional non-improving.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    start = time.time()
    temp = 1.0
    min_temp = 0.01
    cooling = 0.999

    # Build occupied set and pos_to_idx for fast lookup
    def _rebuild(chain):
        occupied = set(chain)
        pos_to_idx = {p: i for i, p in enumerate(chain)}
        return occupied, pos_to_idx

    occupied, pos_to_idx = _rebuild(current_chain)

    while _time_remaining() > time_budget:
        # Pick a random index to try moving
        idx = random.randint(1, n - 2)  # not first or last
        old_pos = current_chain[idx]
        prev_pos = current_chain[idx - 1]
        next_pos = current_chain[idx + 1]

        # Try all 6 neighbours
        candidates = list(_neighbours(old_pos))
        random.shuffle(candidates)

        moved = False
        for new_pos in candidates:
            if new_pos == prev_pos or new_pos == next_pos:
                continue
            if new_pos in occupied:
                continue

            # Remove old, add new
            occupied.remove(old_pos)
            occupied.add(new_pos)
            current_chain[idx] = new_pos

            # Compute energy change
            old_e = _compute_energy(sequence, current_chain)

            if old_e <= current_energy:
                # Accept (improving or equal)
                current_energy = old_e
                pos_to_idx[new_pos] = idx
                del pos_to_idx[old_pos]
                moved = True
                break
            else:
                # Simulated annealing acceptance
                delta = current_energy - old_e  # positive means worse
                if delta < 0:
                    # Actually improving
                    current_energy = old_e
                    pos_to_idx[new_pos] = idx
                    del pos_to_idx[old_pos]
                    moved = True
                    break
                else:
                    # delta > 0, worse
                    if random.random() < math.exp(-delta / temp):
                        current_energy = old_e
                        pos_to_idx[new_pos] = idx
                        del pos_to_idx[old_pos]
                        moved = True
                        break
                    else:
                        # Reject, undo
                        occupied.remove(new_pos)
                        occupied.add(old_pos)
                        current_chain[idx] = old_pos

        if not moved:
            # Try another residue
            pass

        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

        # Cool down
        temp = max(min_temp, temp * cooling)

    return best_chain


import math

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search + local refinement.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    # Phase 1: Beam search
    chain = _beam_search(sequence, dimension, beam_width=50)

    # Phase 2: Local refinement if time permits
    if _time_remaining() > 3.0:
        # Allocate remaining time minus safety margin for refinement
        refine_budget = min(_time_remaining() - 5.0, 10.0)
        if refine_budget > 0.5:
            chain = _local_refine(sequence, chain, time_budget=refine_budget)

    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
