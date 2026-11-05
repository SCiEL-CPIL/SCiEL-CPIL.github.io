"""Construct dense 4-cycle-rainbow edge-coloured graph on 20 vertices.

Combines:
  * fast greedy builder (forbidden-colour computation over length-3 paths)
  * local search with TWO move types:
      - re-color an existing edge (unlocks new additions without losing an edge)
      - remove k edges + re-greedy (escape local optima)
  * fast incremental validity check for the re-colour move
"""
import time
import random
from itertools import combinations

N = 20
N_COLORS = 20
DEADLINE = 132.0  # explicit safety margin before the 150 s kill

# Precompute all unordered pairs once
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]


def _validate(edges):
    """Check that every 4-cycle has four distinct colours."""
    if not edges:
        return True
    adj = {}
    for u, v, c in edges:
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c

    verts = list(range(N))
    for a, b, c, d in combinations(verts, 4):
        cycles = [
            [(a, b), (b, c), (c, d), (d, a)],
            [(a, b), (b, d), (d, c), (c, a)],
            [(a, c), (c, b), (b, d), (d, a)],
        ]
        for cyc in cycles:
            cols = []
            valid_cycle = True
            for (u, v) in cyc:
                if v not in adj.get(u, {}):
                    valid_cycle = False
                    break
                cols.append(adj[u][v])
            if valid_cycle:
                if len(set(cols)) != 4:
                    return False
    return True


def _fast_check_after_change(adj, neighbors, changed_edge):
    """Check only the 4-cycles that involve changed_edge=(u,v).

    Returns True if all such 4-cycles are rainbow (given the current adj).
    A 4-cycle containing (u,v) has the form u - x - y - v - u, i.e.
    x in N(u), y in N(v), x != v, y != u, x != y, and (x,y) is an edge.
    The four edges are (u,x),(x,y),(y,v),(v,u).
    """
    u, v = changed_edge
    for x in list(neighbors[u]):
        if x == v:
            continue
        cx = adj[u][x]
        for y in list(neighbors[v]):
            if y == u or y == x:
                continue
            if adj[x][y] == -1:
                continue
            cols = (cx, adj[x][y], adj[y][v], adj[v][u])
            if len(set(cols)) != 4:
                return False
    return True


def construct():
    start = time.time()

    def elapsed():
        return time.time() - start

    def deadline_ok():
        return elapsed() < DEADLINE

    def greedy_build(seed, initial_edges=None, order_type='random'):
        rng = random.Random(seed)

        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        edges = []

        if initial_edges:
            for u, v, c in initial_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing_set = set((min(u, v), max(u, v)) for u, v, _ in edges)
            remaining = [e for e in ALL_PAIRS if e not in existing_set]
        else:
            remaining = ALL_PAIRS[:]

        if order_type == 'random':
            rng.shuffle(remaining)
        elif order_type == 'reverse':
            remaining.reverse()

        for (u, v) in remaining:
            if v in neighbors[u]:
                continue

            impossible = False
            forbidden = set()

            for a in list(neighbors[u]):
                if a == v:
                    continue
                ca = adj[u][a]
                for b in list(neighbors[v]):
                    if b == u or b == a:
                        continue
                    cb = adj[a][b]
                    if cb == -1:
                        continue
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
                if impossible:
                    break

            if not impossible:
                colors = list(range(N_COLORS))
                rng.shuffle(colors)
                for c in colors:
                    if c not in forbidden:
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        break

        return edges

    best_edges = None
    best_count = 0

    # ---- Phase 1: multiple greedy restarts with different orderings ----
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 120

    for ot in order_types:
        for i in range(num_restarts):
            if not deadline_ok():
                break
            edges = greedy_build(i * 7 + 13, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, v % N_COLORS] for v in range(1, N)]
        best_count = len(best_edges)

    # ---- Phase 2: local search with re-color + remove moves ----
    current_best = best_edges[:]
    refine_iter = 0
    max_refine = 4000

    while deadline_ok() and refine_iter < max_refine:
        refine_iter += 1

        # Build working state from current best
        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        for u, v, c in current_best:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)

        move = random.random()
        candidate = None

        if move < 0.55 and len(current_best) > 0:
            # ---- Re-color move ----
            # pick a random edge
            u, v, c0 = random.choice(current_best)
            tried = random.sample([c for c in range(N_COLORS) if c != c0],
                                  k=min(6, N_COLORS - 1))
            for nc in tried:
                # tentatively change
                adj[u][v] = nc
                adj[v][u] = nc
                if not _fast_check_after_change(adj, neighbors, (u, v)):
                    adj[u][v] = c0
                    adj[v][u] = c0
                    continue
                # valid re-color; build candidate edge list
                cand = [e[:] for e in current_best]
                for e in cand:
                    if (min(e[0], e[1]), max(e[0], e[1])) == (min(u, v), max(u, v)):
                        e[2] = nc
                        break
                # re-greedy to fill gaps unlocked by the re-color
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, cand, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges
                    break
        else:
            # ---- Remove + re-greedy move ----
            k = random.randint(1, 5)
            if len(current_best) > k:
                edges_to_remove = random.sample(current_best, k)
                remove_set = set((min(u, v), max(u, v)) for u, v, _ in edges_to_remove)
                base_edges = [e for e in current_best
                              if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, base_edges, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges

        # Periodically validate the best to be safe (cheap for 20 vertices)
        if refine_iter % 200 == 0 and best_edges is not None:
            if not _validate(best_edges):
                # fall back to a last known-good if we somehow corrupted
                # (should not happen since we only accept validated improvements
                # via greedy, but guard anyway)
                pass

    # Final validation
    if best_edges is not None and not _validate(best_edges):
        return [[0, v, v % N_COLORS] for v in range(1, N)]

    return best_edges if best_edges is not None else [[0, v, v % N_COLORS] for v in range(1, N)]


if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Valid: {_validate(result)}")
