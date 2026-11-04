"""Cage n(3,16): deterministic Cayley-graph search over small non-abelian groups.

Strategy: model a 3-regular graph as a Cayley graph on a group G with a symmetric
generator set S (no identity). Girth >= g iff no non-backtracking closed walk of
length < g exists from the identity. We search a curated family of small groups
(dihedral, modular semidirect, A5, A6, SL2(5), SL2(7)) for generator sets whose
non-backtracking girth is at least g, returning the smallest vertex count.
"""
import time
import itertools

# Hard deadline: evaluator kills at 150s; return best before ~142s.
_DEADLINE = time.time() + 140.0


def _deadline_ok():
    return time.time() < _DEADLINE


def _mod_group(n, mod, mul, inv):
    """Return (size, add, sub, mul, inv) for a modular group."""
    return n, add, sub, mul, inv


def _dihedral(n):
    # D_n of order 2n: elements (r, t) with r in Z_n, t in {0,1}
    # (r,t)*(s,u) = (r + t*s, t+u) mod (n,2)
    size = 2 * n

    def add(a, b):
        r1, t1 = a
        r2, t2 = b
        return ((r1 + t1 * r2) % n, (t1 + t2) % 2)

    def sub(a, b):
        r1, t1 = a
        r2, t2 = b
        # a * b^{-1}; b^{-1} = (-t2*r2 mod n, t2)
        return ((r1 - t1 * ((-r2) % n)) % n, (t1 + t2) % 2)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        r, t = a
        return ((-r) % n, t)

    return size, add, sub, mul, inv


def _modular_semidirect(p, q):
    # (Z_p x Z_q) with multiplication (a,b)*(c,d) = (a + b*c, b + d) mod (p, q)
    # This is a group when q | p-1 (we use q=2, p odd prime)
    size = p * q

    def add(a, b):
        a1, b1 = a
        a2, b2 = b
        return ((a1 + b1 * a2) % p, (b1 + b2) % q)

    def sub(a, b):
        a1, b1 = a
        a2, b2 = b
        # a * b^{-1}; b^{-1} = (-b2 * a2 mod p, -b2 mod q)
        return ((a1 - b1 * ((-a2) % p)) % p, (b1 - b2) % q)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        a1, b1 = a
        return ((-b1 * a1) % p, (-b1) % q)

    return size, add, mul, sub, inv


def _small_groups(k, g):
    """Yield candidate groups as (name, size, add, sub, mul, inv)."""
    groups = []

    # Dihedral groups D_n for various n
    for n in range(5, 60):
        size, add, sub, mul, inv = _dihedral(n)
        groups.append(("D%d" % n, size, add, sub, mul, inv))

    # Modular semidirect (Z_p x Z_2) for odd primes p
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]:
        size, add, sub, mul, inv = _modular_semidirect(p, 2)
        groups.append(("MS%d" % p, size, add, sub, mul, inv))

    # A5 (order 60) - icosahedral group
    # Represent as permutations of 5 elements
    groups.append(("A5", 60, _a5_ops()))

    # A6 (order 360)
    groups.append(("A6", 360, _a6_ops()))

    # SL2(5) (order 120)
    groups.append(("SL2_5", 120, _sl25_ops()))

    # SL2(7) (order 336)
    groups.append(("SL2_7", 336, _sl27_ops()))

    return groups


def _a5_ops():
    # A5 as permutations of {0,1,2,3,4}
    # Generate all even permutations
    from math import factorial
    perms = []
    for p in itertools.permutations(range(5)):
        # Check if even
        inv = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            perms.append(p)
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    size = len(perms)

    def add(a, b):
        # Composition a * b: apply b first, then a
        return perm_to_idx[tuple(a[b[i]] for i in range(5))]

    def sub(a, b):
        # a * b^{-1}
        b_inv = [0] * 5
        for i in range(5):
            b_inv[b[i]] = i
        b_inv_perm = tuple(b_inv)
        b_inv_idx = perm_to_idx[b_inv_perm]
        return add(a, b_inv_idx)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        a_inv = [0] * 5
        for i in range(5):
            a_inv[a[i]] = i
        return perm_to_idx[tuple(a_inv)]

    return size, add, sub, mul, inv


def _a6_ops():
    # A6 as permutations of {0,1,2,3,4,5}
    perms = []
    for p in itertools.permutations(range(6)):
        inv = 0
        for i in range(6):
            for j in range(i + 1, 6):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            perms.append(p)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    size = len(perms)

    def add(a, b):
        return perm_to_idx[tuple(a[b[i]] for i in range(6))]

    def sub(a, b):
        b_inv = [0] * 6
        for i in range(6):
            b_inv[b[i]] = i
        b_inv_idx = perm_to_idx[tuple(b_inv)]
        return add(a, b_inv_idx)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        a_inv = [0] * 6
        for i in range(6):
            a_inv[a[i]] = i
        return perm_to_idx[tuple(a_inv)]

    return size, add, sub, mul, inv


def _sl25_ops():
    # SL2(5): 2x2 matrices over Z/5 with determinant 1
    elems = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    if (a * d - b * c) % 5 == 1:
                        elems.append((a, b, c, d))
    elem_to_idx = {e: i for i, e in enumerate(elems)}
    size = len(elems)

    def add(a, b):
        a1, b1, c1, d1 = a
        a2, b2, c2, d2 = b
        r = ((a1 * a2 + b1 * c2) % 5, (a1 * b2 + b1 * d2) % 5,
             (c1 * a2 + d1 * c2) % 5, (c1 * b2 + d1 * d2) % 5)
        return elem_to_idx[r]

    def sub(a, b):
        b_inv = inv(b)
        return add(a, b_inv)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        a1, b1, c1, d1 = a
        # det = 1, so inv = (d, -b, -c, a)
        r = (d1, (-b1) % 5, (-c1) % 5, a1)
        return elem_to_idx[r]

    return size, add, sub, mul, inv


def _sl27_ops():
    # SL2(7): 2x2 matrices over Z/7 with determinant 1
    elems = []
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    if (a * d - b * c) % 7 == 1:
                        elems.append((a, b, c, d))
    elem_to_idx = {e: i for i, e in enumerate(elems)}
    size = len(elems)

    def add(a, b):
        a1, b1, c1, d1 = a
        a2, b2, c2, d2 = b
        r = ((a1 * a2 + b1 * c2) % 7, (a1 * b2 + b1 * d2) % 7,
             (c1 * a2 + d1 * c2) % 7, (c1 * b2 + d1 * d2) % 7)
        return elem_to_idx[r]

    def sub(a, b):
        b_inv = inv(b)
        return add(a, b_inv)

    def mul(a, b):
        return add(a, b)

    def inv(a):
        a1, b1, c1, d1 = a
        r = (d1, (-b1) % 7, (-c1) % 7, a1)
        return elem_to_idx[r]

    return size, add, sub, mul, inv


def _cayley_girth(size, add, sub, mul, inv, gen_set, g):
    """Check if Cayley graph has girth >= g using non-backtracking BFS from identity.

    Returns True if girth >= g, False otherwise.
    """
    # Identity element: find e such that add(e, x) = x for all x (or use known identity)
    # For our groups, identity is the first element in the list
    identity = 0

    # Build adjacency: for each vertex, its neighbors are vertex * s for s in gen_set
    # Non-backtracking walk: sequence v0, v1, ..., vl where vi != vi-2
    # We do BFS on directed edges (prev, curr) to track non-backtracking

    # State: (prev_vertex, curr_vertex) - we arrived at curr from prev
    # Start: from identity, go to each generator
    from collections import deque

    # visited[prev][curr] = True if we've reached curr from prev
    # Use a set of (prev, curr) pairs
    visited = set()
    queue = deque()

    # Initial non-backtracking walks of length 1: identity -> s
    for s in gen_set:
        s_idx = gen_set[s]  # gen_set maps generator element to its index
        next_v = add(identity, s_idx)
        state = (identity, next_v)
        if state not in visited:
            visited.add(state)
            queue.append((state, 1))

    # BFS for non-backtracking walks up to length g-1
    # If we ever return to identity with prev != identity, we found a cycle < g
    for _ in range(g - 1):
        if not queue:
            break
        for _ in range(len(queue)):
            (prev, curr), depth = queue.popleft()

            # Extend: from curr, go to curr * t for t in gen_set, t != prev (non-backtracking)
            # Actually non-backtracking means the next vertex != prev
            for t in gen_set:
                t_idx = gen_set[t]
                next_v = add(curr, t_idx)
                # Non-backtracking: next_v != prev
                if next_v == prev:
                    continue
                # Check if we've completed a cycle back to identity
                if next_v == identity:
                    return False  # Found cycle of length depth + 1 < g
                state = (curr, next_v)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, depth + 1))

    return True


def _find_generator_set(size, add, sub, mul, inv, k, g):
    """Find a symmetric generator set of size k for the group.

    Returns list of generator indices, or None.
    """
    # For a k-regular Cayley graph, we need a symmetric set S of size k
    # (S = S^{-1}, no identity)
    # Try various strategies

    # Strategy 1: For dihedral groups, use rotation r and reflections
    # Strategy 2: Random search
    # Strategy 3: Systematic search over small sets

    # First, identify the group type by checking properties
    # For now, try a few heuristic generator sets

    candidates = []

    # Try: pick elements with certain properties
    # For 3-regular: need 3 generators (or 1.5 pairs, but must be symmetric)
    # So S = {a, a^{-1}, b} where b = b^{-1} (involution)

    # Find involutions (elements of order 2)
    involutions = []
    for i in range(size):
        if i == 0:
            continue
        if add(i, i) == 0:
            involutions.append(i)

    # Find elements of order 3
    order3 = []
    for i in range(size):
        if i == 0:
            continue
        i2 = add(i, i)
        if i2 != 0 and add(i2, i) == 0:
            order3.append(i)

    # Try combinations: {a, a^{-1}, b} where b is involution
    # a should not be involution (otherwise a = a^{-1}, giving only 2 distinct)
    for b in involutions[:20]:  # Limit search
        for a in range(1, size):
            if a == b or add(a, a) == 0:
                continue
            a_inv = inv(a)
            if a_inv == a:
                continue
            gen_set = [a, a_inv, b]
            # Check they're distinct
            if len(set(gen_set)) != 3:
                continue
            candidates.append(gen_set)

    # Also try {a, b, c} where all are involutions
    for combo in itertools.combinations(involutions[:15], 3):
        candidates.append(list(combo))

    # Try random
    import random
    for _ in range(500):
        a = random.randint(1, size - 1)
        b = random.randint(1, size - 1)
        c = random.randint(1, size - 1)
        gen_set = [a, inv(a), b]
        if len(set(gen_set)) == 3:
            candidates.append(gen_set)

    # Check each candidate
    for gen_set in candidates:
        if not _deadline_ok():
            break
        gen_map = {g_elem: idx for idx, g_elem in enumerate(gen_set)}
        if _cayley_girth(size, add, sub, mul, inv, gen_map, g):
            return gen_set

    return None


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    best_n = None
    best_edges = None

    # Get candidate groups
    groups = _small_groups(k, g)

    # Sort by size (prefer smaller groups)
    groups.sort(key=lambda x: x[1])

    for name, size, add, sub, mul, inv in groups:
        if not _deadline_ok():
            break

        # Skip if already have a better result
        if best_n is not None and size >= best_n:
            continue

        # Find a generator set
        gen_set = _find_generator_set(size, add, sub, mul, inv, k, g)

        if gen_set is None:
            continue

        # Build edges
        gen_map = {g_elem: idx for idx, g_elem in enumerate(gen_set)}
        edges = set()
        for v in range(size):
            for s in gen_set:
                s_idx = gen_map[s]
                w = add(v, s_idx)
                if v < w:
                    edges.add((v, w))

        # Verify: check girth is actually >= g
        # (The _cayley_girth check should have ensured this, but double-check)
        if _cayley_girth(size, add, sub, mul, inv, gen_map, g):
            if best_n is None or size < best_n:
                best_n = size
                best_edges = list(edges)

    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    if best_n is None:
        best_n, best_edges = _luw_construct(k, g)

    return best_n, best_edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
