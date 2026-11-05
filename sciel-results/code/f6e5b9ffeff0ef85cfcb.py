"""Construct a 4-regular graph of girth >= 10 with as few vertices as possible."""
import math
import random
import time
import itertools
from collections import deque

import numpy as np


def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible."""
    if k != 4:
        return _construct_fallback(k, g)

    deadline = time.time() + 110  # safety margin before 120s kill

    # Strategy 1: Search for a 320-vertex Cayley graph with girth >= 10
    # Try many abelian group structures and random generator sets
    result = _search_cayley_320(deadline)
    if result is not None:
        return result

    # Strategy 2: Try non-abelian groups of order 320
    result = _search_nonabelian_320(deadline)
    if result is not None:
        return result

    # Strategy 3: Fallback to LZU-style construction (larger but valid)
    return _construct_fallback(4, g)


def _search_cayley_320(deadline):
    """Search for a 4-regular Cayley graph on an abelian group of order 320
    with girth >= 10. Try many group decompositions and random generators."""
    rng = random.Random(42)

    # Group decompositions of 320 = 2^6 * 5
    decomps = [
        (320,),        # Z_320
        (160, 2),      # Z_160 x Z_2
        (80, 4),       # Z_80 x Z_4
        (40, 8),       # Z_40 x Z_8
        (32, 10),      # Z_32 x Z_10
        (20, 16),      # Z_20 x Z_16
        (16, 20),      # Z_16 x Z_20
        (10, 32),      # Z_10 x Z_32
        (8, 40),       # Z_8 x Z_40
        (4, 80),       # Z_4 x Z_80
        (2, 160),      # Z_2 x Z_160
        (5, 64),       # Z_5 x Z_64
        (64, 5),       # Z_64 x Z_5
        (4, 4, 20),    # Z_4 x Z_4 x Z_20
        (4, 2, 40),    # Z_4 x Z_2 x Z_40
        (2, 2, 80),    # Z_2 x Z_2 x Z_80
        (2, 2, 2, 40), # Z_2^3 x Z_40
        (2, 2, 2, 2, 20),  # Z_2^4 x Z_20
        (2, 2, 2, 2, 2, 5), # Z_2^5 x Z_5
        (4, 4, 2, 10),  # Z_4 x Z_4 x Z_2 x Z_10
        (8, 8, 5),     # Z_8 x Z_8 x Z_5
        (4, 2, 2, 20), # Z_4 x Z_2 x Z_2 x Z_20
        (2, 2, 4, 20), # Z_2 x Z_2 x Z_4 x Z_20
    ]

    # For each decomposition, try random generator pairs
    for moduli in decomps:
        if time.time() > deadline:
            break
        n = 1
        for m in moduli:
            n *= m
        if n != 320:
            continue

        # Number of random attempts per decomposition
        n_attempts = 200 if len(moduli) <= 2 else 50

        for attempt in range(n_attempts):
            if time.time() > deadline:
                break

            # Generate random generator pair (s1, s2) with their inverses
            # s1, s2 must be such that {s1, -s1, s2, -s2} are all distinct
            # and none is the identity
            s1 = tuple(rng.randint(0, m - 1) for m in moduli)
            s2 = tuple(rng.randint(0, m - 1) for m in moduli)

            # Compute inverses
            neg_s1 = tuple((-s1[i]) % moduli[i] for i in range(len(moduli)))
            neg_s2 = tuple((-s2[i]) % moduli[i] for i in range(len(moduli)))

            S = [s1, neg_s1, s2, neg_s2]

            # Check all 4 are distinct and none is identity
            identity = tuple(0 for _ in moduli)
            if len(set(S)) != 4:
                continue
            if identity in S:
                continue
            # Check s1 != -s1 (no element of order 2 for s1)
            if s1 == neg_s1:
                continue
            if s2 == neg_s2:
                continue

            # Quick algebraic check: no short closed walks
            if not _quick_girth_check(moduli, S, 10):
                continue

            # Full girth verification via non-backtracking BFS
            if _full_girth_check(moduli, S, 10):
                # Build edge list
                edges = _build_edges_from_generators(moduli, S)
                return 320, edges

    return None


def _search_nonabelian_320(deadline):
    """Try dihedral and other non-abelian groups of order 320."""
    # D_160: order 320, presentation <r, s | r^160 = 1, s^2 = 1, srs = r^{-1}>
    # Cayley graph with generators {a, a^{-1}, b, b^{-1}} where a, b are group elements
    # This is complex; skip for now as the abelian search should suffice

    # Try Z_160 x Z_2 with a twist (semidirect product)
    # For now, just return None
    return None


def _quick_girth_check(moduli, S, g):
    """Quick algebraic check: enumerate all words of length < g/2 in the generators
    and check if any equals the identity (indicating a short closed walk).
    
    This is a necessary (not sufficient) condition for girth >= g.
    We check words of length up to g//2 = 5.
    """
    n_mods = len(moduli)
    identity = [0] * n_mods

    # BFS over words, tracking positions
    # State: position in the group
    # We want to know if any non-trivial word of length < g evaluates to identity

    # For girth >= g, we need: no word of length 3..g-1 in {s1, -s1, s2, -s2}
    # (with no consecutive inverse pairs) evaluates to identity.
    #
    # Quick check: enumerate all words of length 3, 4, 5 and check if any = identity
    # This is 4*3*3*3*3 = 324 words for length 5, manageable.

    # Actually, let's be smarter: BFS from identity, tracking (position, last_gen_idx)
    # If we reach identity at depth d < g, we have a short cycle.

    # But we must avoid backtracking (consecutive inverse pairs)
    neg_map = {0: 1, 1: 0, 2: 3, 3: 2}

    # visited: set of (pos_tuple, last_idx)
    visited = set()
    # Start: after taking one step in each direction
    for start_idx in range(4):
        pos = [S[start_idx][i] % moduli[i] for i in range(n_mods)]
        state = (tuple(pos), start_idx)
        visited.add(state)

        # Expand up to depth g-1
        frontier = [(pos, start_idx, 1)]
        for depth in range(1, g - 1):
            if time.time() > deadline_global:
                return False  # timeout, be conservative
            new_frontier = []
            for pos, last_idx, d in frontier:
                for next_idx in range(4):
                    if next_idx == neg_map[last_idx]:
                        continue
                    new_pos = [(pos[i] + S[next_idx][i]) % moduli[i] for i in range(n_mods)]
                    new_state = (tuple(new_pos), next_idx)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    # Check if we're back at identity
                    if all(new_pos[i] == 0 for i in range(n_mods)):
                        if d + 1 >= 3:  # cycle length >= 3
                            return False
                    new_frontier.append((new_pos, next_idx, d + 1))
            frontier = new_frontier
            if not frontier:
                break

    return True


def _full_girth_check(moduli, S, g):
    """Full girth check via non-backtracking BFS from every vertex.
    
    For a Cayley graph, by symmetry we only need to check from the identity.
    We do a non-backtracking BFS and check if we return to identity at depth < g.
    """
    n_mods = len(moduli)
    neg_map = {0: 1, 1: 0, 2: 3, 3: 2}

    # Non-backtracking BFS from identity
    # State: (position, last_direction)
    # If we reach identity with depth >= 3, we have a cycle of that length.

    # visited[state] = min depth
    visited = {}

    for start_idx in range(4):
        pos = [S[start_idx][i] % moduli[i] for i in range(n_mods)]
        state = (tuple(pos), start_idx)
        visited[state] = 1
        queue = deque([(pos, start_idx, 1)])

        while queue:
            pos, last_idx, depth = queue.popleft()

            if depth >= g - 1:
                continue

            # Check if back at identity
            if all(pos[i] == 0 for i in range(n_mods)) and depth >= 3:
                return False

            for next_idx in range(4):
                if next_idx == neg_map[last_idx]:
                    continue
                new_pos = [(pos[i] + S[next_idx][i]) % moduli[i] for i in range(n_mods)]
                new_state = (tuple(new_pos), next_idx)
                if new_state not in visited:
                    visited[new_state] = depth + 1
                    queue.append((new_pos, next_idx, depth + 1))

    return True


def _build_edges_from_generators(moduli, S):
    """Build the edge list for a Cayley graph."""
    n_mods = len(moduli)
    n = 1
    for m in moduli:
        n *= m

    # Precompute strides
    strides = [1] * n_mods
    for i in range(1, n_mods):
        strides[i] = strides[i - 1] * moduli[i - 1]

    edge_set = set()
    for idx in range(n):
        # Convert index to group element
        coords = []
        rem = idx
        for i in range(n_mods):
            coords.append(rem % moduli[i])
            rem //= moduli[i]

        for s in S:
            new_coords = [(coords[i] + s[i]) % moduli[i] for i in range(n_mods)]
            # Convert back to index
            new_idx = 0
            for i in range(n_mods):
                new_idx += new_coords[i] * strides[i]

            if idx < new_idx:
                edge_set.add((idx, new_idx))
            else:
                edge_set.add((new_idx, idx))

    return sorted(edge_set)


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar style construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# Global deadline for internal checks
deadline_global = time.time() + 100


def _search_cayley_320(deadline):
    """Search for a 4-regular Cayley graph on an abelian group of order 320
    with girth >= 10. Try many group decompositions and random generators."""
    global deadline_global
    deadline_global = deadline

    rng = random.Random(42)

    # Group decompositions of 320 = 2^6 * 5
    decomps = [
        (320,),
        (160, 2),
        (80, 4),
        (40, 8),
        (32, 10),
        (20, 16),
        (16, 20),
        (10, 32),
        (8, 40),
        (4, 80),
        (2, 160),
        (5, 64),
        (64, 5),
        (4, 4, 20),
        (4, 2, 40),
        (2, 2, 80),
        (2, 2, 2, 40),
        (2, 2, 2, 2, 20),
        (2, 2, 2, 2, 2, 5),
        (4, 4, 2, 10),
        (8, 8, 5),
        (4, 2, 2, 20),
        (2, 2, 4, 20),
    ]

    for moduli in decomps:
        if time.time() > deadline:
            break
        n = 1
        for m in moduli:
            n *= m
        if n != 320:
            continue

        n_attempts = 300 if len(moduli) <= 2 else 80

        for attempt in range(n_attempts):
            if time.time() > deadline:
                break

            s1 = tuple(rng.randint(0, m - 1) for m in moduli)
            s2 = tuple(rng.randint(0, m - 1) for m in moduli)

            neg_s1 = tuple((-s1[i]) % moduli[i] for i in range(len(moduli)))
            neg_s2 = tuple((-s2[i]) % moduli[i] for i in range(len(moduli)))

            S = [s1, neg_s1, s2, neg_s2]

            identity = tuple(0 for _ in moduli)
            if len(set(S)) != 4:
                continue
            if identity in S:
                continue
            if s1 == neg_s1:
                continue
            if s2 == neg_s2:
                continue

            if not _quick_girth_check(moduli, S, 10, deadline):
                continue

            if _full_girth_check(moduli, S, 10, deadline):
                edges = _build_edges_from_generators(moduli, S)
                return 320, edges

    return None


def _quick_girth_check(moduli, S, g, deadline):
    """Quick check: non-backtracking BFS from identity, detect return at depth < g."""
    n_mods = len(moduli)
    neg_map = {0: 1, 1: 0, 2: 3, 3: 2}

    for start_idx in range(4):
        pos = [S[start_idx][i] % moduli[i] for i in range(n_mods)]
        visited = set()
        state = (tuple(pos), start_idx)
        visited.add(state)

        frontier = [(pos, start_idx, 1)]
        for depth in range(1, g - 1):
            if time.time() > deadline:
                return False
            new_frontier = []
            for pos, last_idx, d in frontier:
                for next_idx in range(4):
                    if next_idx == neg_map[last_idx]:
                        continue
                    new_pos = [(pos[i] + S[next_idx][i]) % moduli[i] for i in range(n_mods)]
                    new_state = (tuple(new_pos), next_idx)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if all(new_pos[i] == 0 for i in range(n_mods)) and d + 1 >= 3:
                        return False
                    new_frontier.append((new_pos, next_idx, d + 1))
            frontier = new_frontier
            if not frontier:
                break

    return True


def _full_girth_check(moduli, S, g, deadline):
    """Full girth check via non-backtracking BFS from identity."""
    n_mods = len(moduli)
    neg_map = {0: 1, 1: 0, 2: 3, 3: 2}

    for start_idx in range(4):
        pos = [S[start_idx][i] % moduli[i] for i in range(n_mods)]
        visited = {}
        state = (tuple(pos), start_idx)
        visited[state] = 1
        queue = deque([(pos, start_idx, 1)])

        while queue:
            if time.time() > deadline:
                return False
            pos, last_idx, depth = queue.popleft()

            if depth >= g - 1:
                continue

            if all(pos[i] == 0 for i in range(n_mods)) and depth >= 3:
                return False

            for next_idx in range(4):
                if next_idx == neg_map[last_idx]:
                    continue
                new_pos = [(pos[i] + S[next_idx][i]) % moduli[i] for i in range(n_mods)]
                new_state = (tuple(new_pos), next_idx)
                if new_state not in visited:
                    visited[new_state] = depth + 1
                    queue.append((new_pos, next_idx, depth + 1))

    return True
