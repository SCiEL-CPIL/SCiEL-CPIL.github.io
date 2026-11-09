"""
Maximum clique search for the bounded-lcm graph on [1, 512].
Uses bitset-based Tomita branch-and-bound with greedy coloring,
structural initialization from power-of-2 clique, and polish passes.
"""
import time
import random
from math import gcd

N = 512
DEADLINE = 85.0  # safety margin before 90s kill


def _build_adjacency():
    """
    Build adjacency as bitmasks.
    adj[v] is an integer bitmask where bit u is set iff lcm(v,u) <= 512.
    """
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << b)
        adj[a] = mask
    return adj


def _popcount(x):
    """Fast popcount for Python integers."""
    return bin(x).count('1')


def _greedy_color(adj, P):
    """
    Greedy coloring of the subgraph induced by vertex set P (given as a bitmask).
    Returns (color_class, ordered) where color_class[v] = color index (0-based),
    and ordered is the list of vertices sorted by color (ascending).
    """
    if P == 0:
        return {}, []

    # Extract vertices
    verts = []
    m = P
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb

    # Sort by degree within P descending for better coloring
    verts.sort(key=lambda v: _popcount(adj[v] & P), reverse=True)

    color_class = {}
    color = 0
    remaining = list(verts)

    while remaining:
        color_set = []
        new_remaining = []
        for v in remaining:
            can_add = True
            for u in color_set:
                if adj[v] & (1 << u):
                    can_add = False
                    break
            if can_add:
                color_set.append(v)
            else:
                new_remaining.append(v)

        for v in color_set:
            color_class[v] = color
        color += 1
        remaining = new_remaining

    # Sort by color ascending
    ordered = sorted(verts, key=lambda v: color_class[v])
    return color_class, ordered


def _max_clique(adj, P, current_clique, best_clique, deadline):
    """
    Tomita-style branch and bound for maximum clique.
    P: bitmask of candidate vertices
    current_clique: list of vertices in current clique
    best_clique: list of vertices in best clique found so far (modified in place)
    Returns best_clique.
    """
    if time.time() > deadline:
        return best_clique

    n_P = _popcount(P)
    if n_P == 0:
        return best_clique

    if len(current_clique) + n_P <= len(best_clique):
        return best_clique

    # Greedy coloring
    color_class, ordered = _greedy_color(adj, P)
    num_colors = max(color_class.values()) + 1 if color_class else 0

    # Upper bound check
    if len(current_clique) + num_colors <= len(best_clique):
        return best_clique

    # Process in reverse order (highest color first)
    for i in range(len(ordered) - 1, -1, -1):
        if time.time() > deadline:
            break

        v = ordered[i]

        # Upper bound: len(current) + number of vertices from i onwards
        if len(current_clique) + i + 1 <= len(best_clique):
            break

        # Add v to clique
        current_clique.append(v)

        if len(current_clique) > len(best_clique):
            best_clique.clear()
            best_clique.extend(current_clique)

        # New candidates: vertices in ordered[0:i] that are adjacent to v
        new_P = 0
        for j in range(i):
            u = ordered[j]
            if adj[v] & (1 << u):
                new_P |= (1 << u)

        # Recurse
        _max_clique(adj, new_P, current_clique, best_clique, deadline)

        # Remove v
        current_clique.pop()

    return best_clique


def _greedy_clique(adj, start_mask):
    """
    Greedy heuristic: repeatedly pick the vertex with highest degree within remaining candidates.
    """
    clique = []
    candidates = start_mask

    while candidates:
        # Pick vertex with max degree within candidates
        best_v = -1
        best_deg = -1
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = _popcount(adj[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb

        if best_v < 0:
            break

        clique.append(best_v)
        candidates = candidates & adj[best_v]

    return clique


def _try_extend_clique(adj, clique, deadline):
    """
    Try to extend the clique by finding vertices that are adjacent to all members.
    Uses iterative deepening: try to add one vertex at a time.
    """
    clique_set = set(clique)
    improved = True
    while improved:
        improved = False
        # Find all vertices adjacent to all clique members
        addable = (1 << (N + 1)) - 2  # all bits except bit 0
        for u in clique:
            addable &= adj[u]
        # Remove clique members
        for u in clique:
            addable &= ~(1 << u)

        if addable == 0:
            break

        # Try adding each candidate vertex
        m = addable
        while m:
            if time.time() > deadline:
                return clique
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb

            # Check if v is adjacent to all current clique members
            # (already guaranteed by addable mask, but verify)
            new_clique = clique + [v]
            # Try to extend further from this point
            extended = _greedy_clique(adj, addable & adj[v])
            if extended:
                candidate = new_clique + extended
                if len(candidate) > len(clique):
                    clique = candidate
                    improved = True
                    break

    return clique


def _polish_swap(adj, clique, deadline):
    """
    Try to improve clique by removing one vertex and adding others.
    """
    clique_set = set(clique)
    n_clique = len(clique)

    # Try single swap: remove one, try to add a non-member that's adjacent to all remaining
    for i in range(n_clique):
        if time.time() > deadline:
            return clique, False
        removed = clique[i]
        remaining = [clique[j] for j in range(n_clique) if j != i]
        remaining_set = set(remaining)

        # Find all vertices adjacent to all in remaining
        addable_mask = (1 << (N + 1)) - 2  # all bits set except bit 0
        for u in remaining:
            addable_mask &= adj[u]
        # Remove remaining
        for u in remaining:
            addable_mask &= ~(1 << u)

        new_adds = []
        m = addable_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if v not in clique_set:
                new_adds.append(v)
            m ^= lsb

        if new_adds:
            # Try adding the best one (highest degree in addable set)
            best_new = max(new_adds, key=lambda v: _popcount(adj[v] & addable_mask))
            new_clique = remaining + [best_new]
            if len(new_clique) > n_clique:
                return new_clique, True

    return clique, False


def _structural_init():
    """
    Generate a good initial clique using structural knowledge.
    Powers of 2 form a clique: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    """
    # Powers of 2 up to 512
    powers_of_2 = []
    v = 1
    while v <= N:
        powers_of_2.append(v)
        v *= 2
    return powers_of_2


def construct():
    start_time = time.time()

    # Build adjacency
    adj = _build_adjacency()

    # Full mask of vertices 1..512
    full_mask = 0
    for v in range(1, N + 1):
        full_mask |= (1 << v)

    # Initialize with structural clique (powers of 2)
    best_clique = _structural_init()

    # Try to extend the structural clique
    best_clique = _try_extend_clique(adj, best_clique, DEADLINE - 10)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return list(best_clique)

    # Run Tomita B&B on the full graph, seeded with our best clique
    best_clique_copy = list(best_clique)
    _max_clique(adj, full_mask, [], best_clique_copy, DEADLINE - 20)
    if len(best_clique_copy) > len(best_clique):
        best_clique = best_clique_copy

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        return list(best_clique)

    # Multiple random restarts with greedy + local B&B
    rng = random.Random(42)
    num_restarts = 0
    max_restarts = 300

    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:  # leave 15s for final polish
            break
        if num_restarts >= max_restarts:
            break
        num_restarts += 1

        # Random subset for greedy - bias towards high-degree vertices
        sample_size = min(N, rng.randint(200, 450))
        sample = rng.sample(range(1, N + 1), sample_size)
        sample_mask = 0
        for v in sample:
            sample_mask |= (1 << v)

        greedy_result = _greedy_clique(adj, sample_mask)

        if len(greedy_result) > len(best_clique):
            best_clique = greedy_result
            # Run B&B from this clique
            remaining_mask = 0
            clique_set = set(greedy_result)
            for v in range(1, N + 1):
                if v not in clique_set:
                    ok = True
                    for u in greedy_result:
                        if not (adj[v] & (1 << u)):
                            ok = False
                            break
                    if ok:
                        remaining_mask |= (1 << v)
            if remaining_mask:
                best_clique_copy = list(best_clique)
                _max_clique(adj, remaining_mask, list(greedy_result), best_clique_copy, DEADLINE - 15)
                if len(best_clique_copy) > len(best_clique):
                    best_clique = best_clique_copy
                if time.time() > DEADLINE - 15:
                    break

    # Polish pass - multiple rounds
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 1:
                break
            best_clique, improved = _polish_swap(adj, best_clique, DEADLINE - 1)
            if not improved:
                # Try extending from scratch
                extended = _try_extend_clique(adj, list(best_clique), DEADLINE - 1)
                if len(extended) > len(best_clique):
                    best_clique = extended
                    improved = True
                else:
                    break

    return list(best_clique)
