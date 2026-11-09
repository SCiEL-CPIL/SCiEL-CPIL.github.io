"""Rewritten program: joint SA over values and edges with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 80.0  # wall-clock safety margin (evaluator kills at 90s)

def _score(values, edges):
    """Return max(#distinct sums, #distinct products) over the given edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _initial_solution():
    """A deterministic-ish starting point: small values, random 35 edges."""
    values = list(range(1, 15))
    all_edges = list(combinations(range(14), 2))
    random.shuffle(all_edges)
    edges = [list(e) for e in all_edges[:35]]
    return values, edges

def _swap_values(values, i, j):
    values[i], values[j] = values[j], values[i]

def _random_value(values, idx):
    values[idx] = random.randint(1, 10000)

def _swap_edge(edges, a, b):
    edges[a], edges[b] = edges[b], edges[a]

def _replace_edge(edges, idx, new_edge):
    edges[idx] = list(new_edge)

def construct():
    start = time.time()
    random.seed(12345)  # deterministic for reproducibility; change if needed

    values, edges = _initial_solution()
    best = _score(values, edges)
    best_values = list(values)
    best_edges = [list(e) for e in edges]

    all_edges = list(combinations(range(14), 2))
    all_edges_set = set(all_edges)

    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995

    step = 0
    while time.time() < _DEADLINE:
        step += 1
        # Choose a mutation type
        r = random.random()
        if r < 0.45:
            # Swap two values
            i, j = random.sample(range(14), 2)
            old_i, old_j = values[i], values[j]
            _swap_values(values, i, j)
        elif r < 0.70:
            # Replace one value
            idx = random.randrange(14)
            old = values[idx]
            _random_value(values, idx)
        elif r < 0.85:
            # Swap two edges
            a, b = random.sample(range(35), 2)
            old_a, old_b = edges[a], edges[b]
            _swap_edge(edges, a, b)
        else:
            # Replace one edge with a random valid edge (no loops/dups)
            idx = random.randrange(35)
            old = edges[idx]
            # pick a random edge not currently in the set
            current_set = set(tuple(e) for e in edges)
            for _ in range(20):
                cand = random.choice(all_edges)
                if cand not in current_set:
                    break
            _replace_edge(edges, idx, cand)

        new_score = _score(values, edges)
        if new_score <= best:
            # accept (and possibly improve)
            if new_score < best:
                best = new_score
                best_values = list(values)
                best_edges = [list(e) for e in edges]
                T = max(T_min, T * 1.05)  # slight reheat
        else:
            # probabilistic accept
            import math
            if random.random() < math.exp(-(new_score - best) / max(T, 1e-9)):
                pass  # accept worse
            else:
                # reject: revert
                if r < 0.45:
                    _swap_values(values, i, j)
                elif r < 0.70:
                    values[idx] = old
                elif r < 0.85:
                    _swap_edge(edges, a, b)
                else:
                    edges[idx] = old
        T = max(T_min, T * cooling)

    # Final polish: try to reduce distinct counts via targeted edge swaps
    # (kept simple to stay within time)
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges],
    }
# EVOLVE-BLOCK-END
