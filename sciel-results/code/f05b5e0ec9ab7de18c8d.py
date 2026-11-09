"""Rewritten HP lattice protein folding with pivot-move simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

# 9 non-identity 90-degree rotation matrices about x, y, z axes (3 each).
_ROTATIONS = [
    np.array([[1,0,0],[0,0,-1],[0,1,0]]),
    np.array([[1,0,0],[0,-1,0],[0,-1,0]]),
    np.array([[1,0,0],[0,0,1],[0,-1,0]]),
    np.array([[0,0,1],[0,1,0],[-1,0,0]]),
    np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
    np.array([[0,0,-1],[0,1,0],[1,0,0]]),
    np.array([[0,-1,0],[1,0,0],[0,0,1]]),
    np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
    np.array([[0,1,0],[-1,0,0],[0,0,1]]),
]
ROTS = _ROTATIONS


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy_array(seq_is_h, chain):
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    n = len(seq_is_h)
    contacts = 0
    for i in range(n):
        if not seq_is_h[i]:
            continue
        p = chain[i]
        for ax in range(3):
            for s in (-1, 1):
                q = list(p); q[ax] += s
                j = where.get(tuple(q))
                if j is not None and j > i + 1 and seq_is_h[j]:
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    n = len(sequence)
    seq_is_h = np.array([c == "H" for c in sequence])

    deadline = time.time() + 315.0  # safety margin before 330s wall-clock kill

    rng = random.Random(12345 + n)
    starts = []
    t0 = time.time()
    while time.time() - t0 < 1.0:
        c = _grow(sequence, dimension, rng)
        if c is not None:
            starts.append(c)
        if len(starts) >= 40:
            break
    if not starts:
        starts = [[tuple([i] + [0] * (dimension - 1)) for i in range(n)]]

    best = None
    best_energy = 10**9
    for c in starts:
        e = _energy_array(seq_is_h, np.array(c, dtype=int))
        if e < best_energy:
            best_energy = e
            best = c
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy_array(seq_is_h, np.array(best, dtype=int))

    def _pivot_step(chain_arr, h, rot_idx):
        R = ROTS[rot_idx]
        tail = chain_arr[h + 1:]
        if len(tail) == 0:
            return None
        rel = tail - chain_arr[h]
        new_rel = rel @ R.T
        new_tail = chain_arr[h] + new_rel
        head_set = set(map(tuple, chain_arr[:h + 1]))
        seen = set()
        for p in new_tail:
            t = tuple(p)
            if t in head_set or t in seen:
                return None
            seen.add(t)
        new_chain = np.empty_like(chain_arr)
        new_chain[:h + 1] = chain_arr[:h + 1]
        new_chain[h + 1:] = new_tail
        new_occ = head_set | seen
        return new_chain, new_occ

    def _delta_energy(old_chain, new_chain, h):
        moved = set(range(h + 1, n))
        where_new = {tuple(p): i for i, p in enumerate(new_chain)}
        where_old = {tuple(p): i for i, p in enumerate(old_chain)}
        def _count(chain, where):
            c = 0
            for i in range(n):
                if not seq_is_h[i]:
                    continue
                p = chain[i]
                for ax in range(3):
                    for s in (-1, 1):
                        q = list(p); q[ax] += s
                        j = where.get(tuple(q))
                        if j is not None and j > i + 1 and seq_is_h[j]:
                            if i in moved or j in moved:
                                c += 1
            return -c
        return _count(new_chain, where_new) - _count(old_chain, where_old)

    T0 = 1.0
    Tend = 0.05
    n_restarts = 6
    per_restart = (deadline - time.time()) / max(1, n_restarts)

    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        if r == 0:
            chain = np.array(best, dtype=int)
        else:
            c = rng.choice(starts)
            chain = np.array(c, dtype=int)
        occ = set(map(tuple, chain))
        e = _energy_array(seq_is_h, chain)
        if e < best_energy:
            best_energy = e
            best = [list(p) for p in chain]

        T = T0
        restart_deadline = time.time() + per_restart
        steps = 0
        while time.time() < restart_deadline:
            h = rng.randint(1, n - 2)
            rot_idx = rng.randint(0, len(ROTS) - 1)
            res = _pivot_step(chain, h, rot_idx)
            if res is None:
                continue
            new_chain, new_occ = res
            de = _delta_energy(chain, new_chain, h)
            if de <= 0 or rng.random() < math.exp(-de / T):
                chain, occ, e = new_chain, new_occ, e + de
                if e < best_energy:
                    best_energy = e
                    best = [list(p) for p in chain]
            T = max(Tend, T * 0.99995)
            steps += 1
            if steps % 5000 == 0 and e > best_energy:
                T = min(T0, T * 2.0)

    return best
# EVOLVE-BLOCK-END
