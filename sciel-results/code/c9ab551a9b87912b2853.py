"""HP lattice folding: fast chain growth + true-energy 2-opt local search."""
# EVOLVE-BLOCK-START
import random
import time

# Precompute neighbour deltas (square lattice, dimension 2)
_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Minus the number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        x, y = p
        for dx, dy in _DELTAS:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, start):
    """One randomized chain growth: prefer sites with new H-H contacts, then free space."""
    chain, occupied = [start], {start: 0}
    last = start
    for i in range(1, len(sequence)):
        x, y = last
        is_h = sequence[i] == "H"
        best_key = None
        best_site = None
        for dx, dy in _DELTAS:
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if is_h:
                sx, sy = site
                # neighbours of this candidate site
                for dx2, dy2 in _DELTAS:
                    q = (sx + dx2, sy + dy2)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy = site
            for dx2, dy2 in _DELTAS:
                if (sx + dx2, sy + dy2) not in occupied:
                    free += 1
            key = gain * 3.0 + free * 0.1 + rng.random()
            if best_key is None or key > best_key:
                best_key = key
                best_site = site
        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
        last = best_site
    return chain


def _local_search(sequence, chain, rng, max_passes=4):
    """Improve a valid fold by relocating single residues to free sites (true-energy greedy)."""
    n = len(chain)
    energy = _energy(sequence, chain)
    for _ in range(max_passes):
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        occupied = {p: i for i, p in enumerate(chain)}
        for i in order:
            if sequence[i] != "H":
                continue
            cur = chain[i]
            # candidate target sites: neighbours of H residues not already adjacent to cur
            candidates = set()
            for dx, dy in _DELTAS:
                x, y = cur
                for dx2, dy2 in _DELTAS:
                    q = (x + dx2, y + dy2)
                    # only consider sites that could form a contact with an H residue
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(j - i) > 1:
                        candidates.add(q)
            # also a few random free sites to escape plateaus
            if rng.random() < 0.5:
                for _ in range(2):
                    x, y = cur
                    for dx2, dy2 in _DELTAS:
                        q = (x + dx2, y + dy2)
                        if q not in occupied:
                            candidates.add(q)
                            break
            for t in candidates:
                if t == cur:
                    continue
                if t in occupied:
                    continue
                # check the move keeps a valid self-avoiding walk with unit steps
                # neighbours in chain: i-1 and i+1 must be adjacent to t
                ok = True
                if i > 0:
                    a = chain[i - 1]
                    if abs(a[0] - t[0]) + abs(a[1] - t[1]) != 1:
                        ok = False
                if ok and i < n - 1:
                    b = chain[i + 1]
                    if abs(b[0] - t[0]) + abs(b[1] - t[1]) != 1:
                        ok = False
                if not ok:
                    continue
                # also ensure t is not the current position of a neighbour (it isn't, unit step)
                # apply move
                occupied.pop(cur)
                occupied[t] = i
                chain[i] = t
                new_energy = _energy(sequence, chain)
                if new_energy < energy:
                    energy = new_energy
                    improved = True
                    break
                else:
                    # revert
                    occupied.pop(t)
                    occupied[cur] = i
                    chain[i] = cur
            # if improved, rebuild occupied (already consistent) and continue
        if not improved:
            break
    return chain, energy


def fold(sequence, dimension):
    """Best conformation from many randomized growths each followed by local search."""
    n = len(sequence)
    deadline = time.time() + 105.0  # safety margin inside 180s budget
    rng = random.Random(12345 + n)

    best = None
    best_energy = 1  # energies are negative

    # initial straight line as guaranteed fallback
    straight = [(i, 0) for i in range(n)]

    # number of restarts: scale with length but bounded by time
    n_restarts = 300
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        # vary the start and the growth noise
        start = (0, 0)
        chain = _grow(sequence, dimension, rng, start)
        if chain is None:
            continue
        # local search on this fold
        chain2, e = _local_search(sequence, chain, rng, max_passes=3)
        if e < best_energy:
            best, best_energy = chain2, e

    if best is None:
        best = straight
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
