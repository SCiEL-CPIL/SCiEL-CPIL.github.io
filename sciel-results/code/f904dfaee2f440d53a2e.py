"""Rewritten HP lattice protein folder: Monte Carlo + true 3D pivots + incremental energy."""
# EVOLVE-BLOCK-START
import time
import math
import random


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


_ROT_X = [
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]
_ROT_Y = [
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
]
_ROT_Z = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
]
_ROTATIONS = (_ROT_X, _ROT_Y, _ROT_Z)


def _apply_rotation(vec, mat):
    return (
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    )


# ---------------------------------------------------------------------------
# Energy (reference, recomputed from scratch)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Initial conformation
# ---------------------------------------------------------------------------
def _straight(n):
    return [(i, 0, 0) for i in range(n)]


def _initial_walk(n, rng):
    if n <= 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    directions = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    ]
    helix_dirs = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    hi = 0
    cur = (0, 0, 0)
    for step in range(1, n):
        placed = False
        order = [helix_dirs[hi % 4]] + [d for d in directions if d != helix_dirs[hi % 4]]
        for d in order:
            nxt = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if nxt not in occupied:
                occupied.add(nxt)
                chain.append(nxt)
                cur = nxt
                placed = True
                break
        if not placed:
            for dz in (1, -1):
                nxt = (cur[0], cur[1], cur[2] + dz)
                if nxt not in occupied:
                    occupied.add(nxt)
                    chain.append(nxt)
                    cur = nxt
                    placed = True
                    break
        if not placed:
            z = 1
            while (cur[0] + 1, cur[1], cur[2] + z) in occupied:
                z += 1
            nxt = (cur[0] + 1, cur[1], cur[2] + z)
            occupied.add(nxt)
            chain.append(nxt)
            cur = nxt
        hi += 1
    return chain


# ---------------------------------------------------------------------------
# Monte Carlo with pivots and incremental energy
# ---------------------------------------------------------------------------
def _mc_fold(sequence, n, rng, deadline):
    chain = _initial_walk(n, rng)
    if len(chain) != n:
        chain = _straight(n)
    is_h = [c == "H" for c in sequence]

    contacts = set()
    pos_map = {p: i for i, p in enumerate(chain)}
    for i in range(n):
        if not is_h[i]:
            continue
        for q in _neighbours(chain[i]):
            j = pos_map.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts.add((i, j))
    energy = -len(contacts)

    best_chain = list(chain)
    best_energy = energy

    t0 = 0.6
    tmin = 0.02
    iters = 15000
    cool = (tmin / t0) ** (1.0 / max(1, iters))
    temp = t0

    for _ in range(iters):
        if time.time() >= deadline:
            break
        if n < 2:
            break

        p = rng.randrange(n - 1)
        max_len = n - 1 - p
        L = rng.randrange(1, max_len + 1)
        L = min(L, max_len)

        a = chain[p]
        b = chain[p + 1]

        for _ in range(3):
            mat = _ROTATIONS[rng.randrange(3)]
            new_block = []
            for idx in range(p + 1, p + 1 + L):
                pos = chain[idx]
                rel = (pos[0] - a[0], pos[1] - a[1], pos[2] - a[2])
                nr = _apply_rotation(rel, mat)
                new_block.append((a[0] + nr[0], a[1] + nr[1], a[2] + nr[2]))

            if new_block[0] != b:
                continue

            fixed_set = set()
            for idx in range(n):
                if idx <= p or idx > p + L:
                    fixed_set.add(chain[idx])

            block_set = set()
            ok = True
            for pos in new_block:
                if pos in fixed_set or pos in block_set:
                    ok = False
                    break
                block_set.add(pos)
            if not ok:
                continue

            block_indices = set(range(p + 1, p + 1 + L))
            new_contacts = set()
            for (i, j) in contacts:
                if i not in block_indices and j not in block_indices:
                    new_contacts.add((i, j))

            new_pos = {}
            block_pos_to_idx = {}
            for k, idx in enumerate(range(p + 1, p + 1 + L)):
                new_pos[idx] = new_block[k]
                block_pos_to_idx[new_block[k]] = idx

            site_to_idx = {}
            for idx in range(n):
                if idx <= p or idx > p + L:
                    site_to_idx[chain[idx]] = idx

            for idx in block_indices:
                if not is_h[idx]:
                    continue
                for q in _neighbours(new_pos[idx]):
                    j = site_to_idx.get(q)
                    if j is not None and abs(j - idx) > 1 and is_h[j]:
                        new_contacts.add((min(idx, j), max(idx, j)))
                    else:
                        j2 = block_pos_to_idx.get(q)
                        if j2 is not None and j2 > idx and abs(j2 - idx) > 1 and is_h[j2]:
                            new_contacts.add((idx, j2))

            new_energy = -len(new_contacts)
            delta = new_energy - energy

            if delta < 0 or (temp > 0 and rng.random() < math.exp(delta / temp)):
                for k, idx in enumerate(range(p + 1, p + 1 + L)):
                    chain[idx] = new_block[k]
                pos_map = {pos: i for i, pos in enumerate(chain)}
                contacts = new_contacts
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                break

        temp *= cool

    return best_chain, best_energy


def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    rng = random.Random(12345 + n)
    deadline = time.time() + 290.0

    best_chain = _straight(n)
    best_energy = _energy(sequence, best_chain)

    while time.time() < deadline - 5.0:
        slice_end = time.time() + min(20.0, max(2.0, 0.05 * n))
        if slice_end > deadline - 5.0:
            slice_end = deadline - 5.0
        chain, energy = _mc_fold(sequence, n, rng, slice_end)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        if time.time() >= deadline - 5.0:
            break

    if len(best_chain) != n or len(set(best_chain)) != n:
        best_chain = _straight(n)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
