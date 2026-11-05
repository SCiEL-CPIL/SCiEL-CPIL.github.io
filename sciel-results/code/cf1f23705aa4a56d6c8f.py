"""Rewritten program: builder + local-search refinement with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 300.0  # safety margin below the 330s kill
_START = None


def _now():
    return time.monotonic()


def _time_left():
    global _START
    if _START is None:
        _START = _now()
    return _DEADLINE - (_now() - _START)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Greedy chain growth with look-ahead: immediate H-H gain + potential future H-H gain."""
    start = (0,) * dimension
    chain, occupied = [start], {start}
    h_positions = set()
    for i in range(1, len(sequence)):
        is_h = sequence[i] == "H"
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_h:
                # immediate H-H contacts
                for q in _neighbours(site):
                    if q in h_positions:
                        gain += 1
                # potential future H-H: count P/H neighbours that are not yet placed
                # (approximation: neighbours that are free and could become H)
                for q in _neighbours(site):
                    if q not in occupied:
                        gain += 0.25
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied.add(site)
        chain.append(site)
        if is_h:
            h_positions.add(site)
    return chain


def _refine(sequence, chain, budget):
    """Local search: single-residue moves and swaps, accept improvements."""
    dim = len(chain[0])
    n = len(chain)
    energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(chain)}
    improved = True
    while improved and _time_left() > 0.5:
        improved = False
        # single-residue moves
        for i in range(n):
            if _time_left() < 0.2:
                break
            old = chain[i]
            for site in _neighbours(old):
                if site in occupied:
                    continue
                # check adjacency constraint: site must not be adjacent to chain[i-1] or chain[i+1]
                # (those are the only neighbours along the chain)
                if i > 0:
                    d = [abs(site[k] - chain[i-1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        continue
                if i < n - 1:
                    d = [abs(site[k] - chain[i+1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        continue
                # tentatively move
                chain[i] = site
                occupied[site] = i
                del occupied[old]
                new_e = _energy(sequence, chain)
                if new_e < energy:
                    energy = new_e
                    improved = True
                else:
                    # revert
                    chain[i] = old
                    occupied[old] = i
                    del occupied[site]
        # swaps
        for i in range(n):
            if _time_left() < 0.2:
                break
            for j in range(i + 2, n):
                if _time_left() < 0.1:
                    break
                if j == i + 1:
                    continue
                pi, pj = chain[i], chain[j]
                # swap: check adjacency for both
                ok = True
                if i > 0:
                    d = [abs(pj[k] - chain[i-1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        ok = False
                if ok and i < n - 1:
                    d = [abs(pj[k] - chain[i+1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        ok = False
                if ok and j > 0:
                    d = [abs(pi[k] - chain[j-1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        ok = False
                if ok and j < n - 1:
                    d = [abs(pi[k] - chain[j+1][k]) for k in range(dim)]
                    if sum(d) == 1:
                        ok = False
                if not ok:
                    continue
                chain[i], chain[j] = pj, pi
                occupied[pj] = i
                occupied[pi] = j
                new_e = _energy(sequence, chain)
                if new_e < energy:
                    energy = new_e
                    improved = True
                else:
                    chain[i], chain[j] = pi, pj
                    occupied[pi] = i
                    occupied[pj] = j
    return chain, energy


def fold(sequence, dimension):
    global _START
    if _START is None:
        _START = _now()
    rng = random.Random(98765 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)
    # number of builder restarts: scale with time and length
    restarts = 30
    for r in range(restarts):
        if _time_left() < 2.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # refine
        chain, energy = _refine(sequence, chain, _time_left())
        if energy < best_energy:
            best, best_energy = chain, energy
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
