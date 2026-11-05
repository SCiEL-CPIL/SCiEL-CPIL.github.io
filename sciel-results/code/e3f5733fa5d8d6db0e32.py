import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0


def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)


def _build_factors():
    """Round-robin tournament: 15 perfect matchings of K_16."""
    n = N
    fixed = n - 1
    rot = list(range(n - 1))
    factors = []
    for r in range(n - 1):
        matching = []
        matching.append((fixed, rot[0]))
        for i in range(1, (n - 1) // 2 + 1):
            a = rot[i]
            b = rot[n - 1 - i]
            matching.append((a, b))
        factors.append(matching)
        rot = rot[-1] + rot[:-1]
    return factors


def _precompute_cycles():
    """Precompute all 4-cycles and edge-to-cycle mapping."""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(d, a)))
        cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(d, c), _edge_index(c, a)))
        cycles.append((_edge_index(a, c), _edge_index(c, b), _edge_index(b, d), _edge_index(d, a)))
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return cycles, edge_to_cycles


def _check_single(cycles, colors, edge_idx, new_color, edge_to_cycles):
    """Check if changing edge_idx to new_color violates any 4-cycle."""
    for ci in edge_to_cycles[edge_idx]:
        cycle = cycles[ci]
        cs = tuple(colors[ei] if ei != edge_idx else new_color for ei in cycle)
        if len(set(cs)) == 4:
            return False
    return True


def construct():
    t0 = time.time()
    factors = _build_factors()
    cycles, edge_to_cycles = _precompute_cycles()

    # Initialize with 1-factorization: 15 colors
    colors = [0] * NUM_EDGES
    for fi, factor in enumerate(factors):
        for (u, v) in factor:
            colors[_edge_index(u, v)] = fi

    next_color = len(factors)
    rng = random.Random(42)

    # Greedily try to assign new colors to edges
    for _ in range(200):
        if time.time() - t0 > DEADLINE - 5:
            break
        order = list(range(NUM_EDGES))
        rng.shuffle(order)
        any_change = False
        for ei in order:
            if time.time() - t0 > DEADLINE - 3:
                break
            # Try the next new color
            c = next_color
            if _check_single(cycles, colors, ei, c, edge_to_cycles):
                colors[ei] = c
                next_color += 1
                any_change = True
            else:
                # Try a few other new colors
                for c2 in range(next_color, next_color + 10):
                    if _check_single(cycles, colors, ei, c2, edge_to_cycles):
                        colors[ei] = c2
                        next_color = max(next_color, c2 + 1)
                        any_change = True
                        break
        if not any_change:
            break

    return colors
