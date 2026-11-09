"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Returns delta_e (new_energy - old_energy).
    """
    delta = 0
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta -= 1
    return delta


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move (pivot). Returns (new_chain, new_where, new_energy, accepted).
    Uses incremental energy delta for speed.
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        new_energy = current_energy + delta_e
        return chain, where, new_energy, True
    else:
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    chain[i:j + 1] = chain[i:j + 1][::-1]
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = chain[i:j + 1][::-1]
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _rotate_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a 90-degree rotation of a segment. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    pivot = chain[i]
    seg = chain[i:j + 1]

    rotated_seg = [pivot]
    for pt in seg[1:]:
        dx = pt[0] - pivot[0]
        dy = pt[1] - pivot[1]
        new_pt = (pivot[0] + dy, pivot[1] - dx)
        rotated_seg.append(new_pt)

    new_where = dict(where)
    for pt in seg[1:]:
        if pt in new_where:
            del new_where[pt]

    for pt in rotated_seg[1:]:
        if pt in new_where:
            return chain, where, current_energy, False

    if j < n - 1:
        if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    chain[i:j + 1] = rotated_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0, seed=12345):
    """Simulated annealing with local moves, reversals, and rotations."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 3.0
    T_min = 0.005
    cooling = 0.999
    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 5000

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.45:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.75:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            T *= 0.95
            no_improve_count = 0

        T *= cooling

    return best_chain, best_energy


def _polish(sequence, chain, max_time=3.0, seed=77777):
    """
    High-precision polish pass: greedy local search at very low temperature
    to find the exact local optimum around the best conformation.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for idx in range(1, n - 1):
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]
            for site in _neighbours(old_site):
                if site == old_site:
                    continue
                if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
                    continue
                if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
                    continue
                if site in where:
                    occ_idx = where[site]
                    if occ_idx != idx:
                        continue
                delta_e = _energy_delta(sequence, chain, idx, old_site, site, where)
                if delta_e < 0:
                    del where[old_site]
                    where[site] = idx
                    chain[idx] = site
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j - i > 5:
                    break
                if i > 0:
                    if abs(chain[i-1][0] - chain[j][0]) + abs(chain[i-1][1] - chain[j][1]) != 1:
                        continue
                if j < n - 1:
                    if abs(chain[i][0] - chain[j+1][0]) + abs(chain[i][1] - chain[j+1][1]) != 1:
                        continue
                seg = chain[i:j+1]
                new_seg = seg[::-1]
                for pt in seg:
                    if where.get(pt) is not None and i <= where[pt] <= j:
                        del where[pt]
                collision = False
                for pt in new_seg:
                    if pt in where:
                        collision = True
                        break
                if collision:
                    for k in range(i, j+1):
                        where[chain[k]] = k
                    continue
                old_e = 0
                for k in range(i, j+1):
                    if sequence[k] == 'H':
                        for q in _neighbours(chain[k]):
                            jj = where.get(q)
                            if jj is not None and abs(jj - k) > 1 and sequence[jj] == 'H' and jj > k:
                                old_e += 1
                for k in range(i, j+1):
                    where[new_seg[k-i]] = k
                new_e = 0
                for k in range(i, j+1):
                    if sequence[k] == 'H':
                        for q in _neighbours(new_seg[k-i]):
                            jj = where.get(q)
                            if jj is not None and abs(jj - k) > 1 and sequence[jj] == 'H' and jj > k:
                                new_e += 1
                delta_e = new_e - old_e
                if delta_e < 0:
                    for k in range(i, j+1):
                        chain[k] = new_seg[k-i]
                        where[chain[k]] = k
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    for k in range(i, j+1):
                        where[chain[k]] = k
    return best_chain, best_energy


def _anneal_restart(sequence, dimension, chain, max_time=10.0, seed=12345, restarts=3):
    """
    Annealing with periodic restarts from best found so far.
    When stuck, restarts from best conformation with fresh high temperature.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    start = time.time()
    restart_count = 0

    while (time.time() - start) < max_time and restart_count < restarts:
        T = 2.0 + 0.5 * restart_count
        T_min = 0.01
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 3000

        while T > T_min and (time.time() - start) < max_time:
            r = rng.random()
            if r < 0.45:
                chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
            elif r < 0.75:
                chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
            else:
                chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

            if accepted:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            if no_improve_count > max_no_improve:
                break

            T *= cooling

        restart_count += 1
        if (time.time() - start) < max_time:
            chain = list(best_chain)
            where = {p: i for i, p in enumerate(chain)}
            current_energy = best_energy

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies, annealing with restarts, and polishing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 400
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8, 0.6, 1.4, 0.9]
    for trial in range(growth_budget):
        if _remaining_time() < 25:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation with restarts
    time_per_anneal = max(0.3, _remaining_time() / max(1, len(initial_chains)) * 0.85)

    initial_chains.sort(key=lambda c: _energy(sequence, c))

    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 15:
            break
        if idx < 3:
            t = time_per_anneal * 2.5
        elif idx < 10:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 8)
        if t <= 0:
            break
        annealed_chain, annealed_energy = _anneal_restart(sequence, dimension, chain, max_time=t, seed=12345 + idx, restarts=3)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Phase 3: High-precision polish
    if _remaining_time() > 8 and best is not None:
        polish_time = min(8.0, _remaining_time() - 5)
        best, best_energy = _polish(sequence, best, max_time=polish_time, seed=99999)

    # Phase 4: If time remains, do one more anneal pass on the polished result
    if _remaining_time() > 12 and best is not None:
        extra_time = min(6.0, _remaining_time() - 6)
        best, best_energy = _anneal_restart(sequence, dimension, best, max_time=extra_time, seed=55555, restarts=2)
        if _remaining_time() > 6:
            final_polish_time = min(3.0, _remaining_time() - 3)
            best, best_energy = _polish(sequence, best, max_time=final_polish_time, seed=88888)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
