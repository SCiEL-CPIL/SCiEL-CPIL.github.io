"""HP lattice protein folding on the cubic lattice.

New design: multi-replica simulated annealing over explicit lattice walks,
starting from a compactness-biased initial conformation and using a 2-opt
(reversal) move that keeps the chain self-avoiding while reshaping long
segments to expose new hydrophobic contacts.
"""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

# Global wall-clock deadline (hard limit 330s; keep a safety margin).
_DEADLINE = time.time() + 322.0

_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _energy(sequence, chain):
    """Negative count of non-adjacent H-H pairs on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for d in _AXES:
                j = where.get((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Compactness-biased self-avoiding walk: prefer directions that move the
    growing tip back toward the origin (tighter packing)."""
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    n = len(sequence)
    for _ in range(1, n):
        tip = chain[-1]
        cand = []
        for d in _AXES:
            q = (tip[0] + d[0], tip[1] + d[1], tip[2] + d[2])
            if q not in occupied:
                cand.append(q)
        if not cand:
            return None
        # Bias toward the centre of mass of the placed part (compactness).
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)
        cz = sum(p[2] for p in chain) / len(chain)
        best = min(cand, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2 + (q[2] - cz) ** 2)
        if rng.random() < 0.75:
            pick = best
        else:
            pick = rng.choice(cand)
        chain.append(pick)
        occupied.add(pick)
    return chain


def _reverse(chain, i, j):
    """Return a new chain with segment [i..j] reversed, or None if it collides."""
    if j <= i:
        return None
    seg = chain[i:j + 1]
    rev = seg[::-1]
    # Determine which sites leave and which are re-occupied.
    leaving = set()
    for p in seg:
        leaving.add(p)
    for p in rev:
        if p not in leaving:
            return None  # would collide with a site that stays occupied
    # Build new list.
    new = chain[:i] + rev + chain[j + 1:]
    return new


def _anneal(sequence, chain0, rng, budget):
    """Simulated annealing with 2-opt moves; returns best chain + energy."""
    n = len(sequence)
    chain = list(chain0)
    best = list(chain)
    best_e = _energy(sequence, chain)
    cur_e = best_e
    end = time.time() + budget
    t = 1.0
    tmin = 0.001
    cooling = 0.9995
    while time.time() < end:
        # pick a reversal segment
        length = rng.randint(3, max(4, n // 4))
        i = rng.randint(0, max(0, n - 1 - length))
        j = i + length
        new = _reverse(chain, i, j)
        if new is None:
            # occasionally try a smaller move
            if n > 6:
                length = rng.randint(2, 5)
                i = rng.randint(0, max(0, n - 1 - length))
                j = i + length
                new = _reverse(chain, i, j)
            if new is None:
                continue
        new_e = _energy(sequence, new)
        delta = new_e - cur_e
        if delta < 0 or rng.random() < math.exp(-delta / t):
            chain = new
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best = list(chain)
        t = max(tmin, t * cooling)
    return best, best_e


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    rng = random.Random(12345 + len(sequence) * 7)
    best, best_e = None, 1
    # Time budget per replica shrinks for long chains.
    if n <= 48:
        per_replica = 1.4
        n_replicas = 6
    elif n <= 96:
        per_replica = 1.1
        n_replicas = 4
    else:
        per_replica = 0.9
        n_replicas = 3
    for r in range(n_replicas):
        if time.time() > _DEADLINE:
            break
        chain0 = _random_walk(sequence, dimension, rng)
        if chain0 is None:
            chain0 = [(i, 0, 0) for i in range(n)]
        c, e = _anneal(sequence, chain0, rng, per_replica)
        if e < best_e:
            best_e = e
            best = c
    if best is None:
        best = [(i, 0, 0) for i in range(n)]
    return [list(p) for p in best]

# EVOLVE-BLOCK-END
