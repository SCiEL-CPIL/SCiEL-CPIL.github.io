"""Decompose each fixed instance graph into few cycles + single edges (Erdos-Gallai)."""

# EVOLVE-BLOCK-START
import time
import random

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline, node_budget=300000):
    """Extract longest cycles greedily; leftover edges become single-edge pieces."""
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    for x in adj:
        adj[x] = list(adj[x])

    remaining = set(edges)
    pieces = []

    def find_longest_cycle():
        best = None
        start = time.time()
        nodes_visited = 0
        # Start from highest-degree vertices (more cycles likely).
        starts = sorted(adj.keys(), key=lambda x: -len(adj[x]))
        for s in starts:
            if time.time() > deadline:
                break
            # BFS/DFS iterative for a simple cycle through s.
            # We look for a cycle by DFS from each neighbor of s.
            for nb in list(adj[s]):
                if (min(s, nb), max(s, nb)) not in remaining:
                    continue
                # DFS from nb, must return to s, simple path.
                stack = [(nb, [nb], {nb})]
                found = None
                while stack and found is None:
                    if time.time() > deadline:
                        return best
                    nodes_visited += 1
                    if nodes_visited > node_budget:
                        break
                    cur, path, used = stack.pop()
                    if cur == s:
                        if len(path) >= 3:
                            found = path
                        continue
                    for w in adj[cur]:
                        if w == s and len(path) >= 2:
                            if (min(cur, s), max(cur, s)) in remaining:
                                found = path + [s]
                                break
                        if w in used:
                            continue
                        if (min(cur, w), max(cur, w)) in remaining:
                            stack.append((w, path + [w], used | {w}))
                if found:
                    if best is None or len(found) > len(best):
                        best = found
                    if len(best) >= n:  # Hamiltonian-ish, stop
                        break
        return best

    while True:
        if time.time() > deadline:
            break
        if not remaining:
            break
        cyc = find_longest_cycle()
        if cyc is None or len(cyc) < 3:
            break
        # Remove cycle edges from remaining.
        cyc_edges = []
        for i in range(len(cyc) - 1):
            a, b = cyc[i], cyc[i + 1]
            e = (min(a, b), max(a, b))
            if e in remaining:
                remaining.remove(e)
                cyc_edges.append(e)
        # closing edge
        a, b = cyc[-1], cyc[0]
        e = (min(a, b), max(a, b))
        if e in remaining:
            remaining.remove(e)
            cyc_edges.append(e)
        if len(cyc_edges) >= 3:
            pieces.append(cyc)

    # leftover edges
    for e in remaining:
        pieces.append([e[0], e[1]])
    return pieces


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s kill
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = _decompose(edges, deadline)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
