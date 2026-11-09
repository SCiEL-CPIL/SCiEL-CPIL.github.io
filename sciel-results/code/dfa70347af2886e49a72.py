import time
import random
from collections import deque

_N = 20
_COLORS = 20
_DEADLINE = 140.0  # safety margin under 150s

def construct():
    start = time.time()
    
    def now_ok():
        return (time.time() - start) < _DEADLINE

    # Precompute all 4-cycles as frozensets of edge tuples
    # A 4-cycle is a subgraph with 4 distinct vertices and 4 edges forming a cycle
    # For 4 distinct vertices {a,b,c,d}, there are 3 possible cycles:
    #   a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    def edge(u, v):
        return (min(u, v), max(u, v))
    
    all_4cycles = []
    for a in range(_N):
        for b in range(a+1, _N):
            for c in range(b+1, _N):
                for d in range(c+1, _N):
                    # Cycle 1: a-b-c-d-a
                    all_4cycles.append(frozenset([edge(a,b), edge(b,c), edge(c,d), edge(d,a)]))
                    # Cycle 2: a-b-d-c-a
                    all_4cycles.append(frozenset([edge(a,b), edge(b,d), edge(d,c), edge(c,a)]))
                    # Cycle 3: a-c-b-d-a
                    all_4cycles.append(frozenset([edge(a,c), edge(c,b), edge(b,d), edge(d,a)]))
    
    all_pairs = [(u, v) for u in range(_N) for v in range(u+1, _N)]
    pair_set = set(all_pairs)
    
    # For each edge, precompute which 4-cycles it participates in
    edge_to_cycles = {p: [] for p in all_pairs}
    for i, cyc in enumerate(all_4cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)
    
    def evaluate_validity(colmat, adj):
        """Check if the current coloring is valid (all 4-cycles are rainbow)."""
        for cyc in all_4cycles:
            # Check if all 4 edges in this cycle are present
            present = True
            for e in cyc:
                if e not in adj or colmat[e[0]][e[1]] is None:
                    present = False
                    break
            if present:
                colors = set()
                for e in cyc:
                    colors.add(colmat[e[0]][e[1]])
                if len(colors) != 4:
                    return False
        return True

    def evaluate_add(colmat, adj, u, v, c):
        """Check if adding edge (u,v) with color c keeps all 4-cycles rainbow.
        Only need to check 4-cycles that include edge (u,v)."""
        e = edge(u, v)
        for cyc_idx in edge_to_cycles[e]:
            cyc = all_4cycles[cyc_idx]
            # Check if all other 3 edges in this cycle are present
            other_edges = [e2 for e2 in cyc if e2 != e]
            all_present = True
            for e2 in other_edges:
                if e2 not in adj or colmat[e2[0]][e2[1]] is None:
                    all_present = False
                    break
            if not all_present:
                continue
            # All 4 edges present, check colors
            colors = {c}
            for e2 in other_edges:
                colors.add(colmat[e2[0]][e2[1]])
            if len(colors) != 4:
                return False
        return True

    def try_build(order, rng=None):
        colmat = [[None]*_N for _ in range(_N)]
        adj = set()
        count = 0
        for (u, v) in order:
            if not now_ok():
                break
            if (min(u,v), max(u,v)) in adj:
                continue
            for c in range(_COLORS):
                if evaluate_add(colmat, adj, u, v, c):
                    colmat[u][v] = c
                    colmat[v][u] = c
                    adj.add(edge(u, v))
                    count += 1
                    break
        edges = []
        for (u, v) in sorted(adj):
            edges.append([u, v, colmat[u][v]])
        return edges

    best = None
    best_count = -1

    rng = random.Random(42)
    
    # Try multiple random orderings
    for trial in range(12):
        if not now_ok():
            break
        o = list(all_pairs)
        rng.shuffle(o)
        edges = try_build(o)
        if len(edges) > best_count:
            best_count = len(edges)
            best = edges
            # Try to add more edges
            if now_ok():
                colmat = [[None]*_N for _ in range(_N)]
                adj = set()
                for (u, v, c) in best:
                    colmat[u][v] = c
                    colmat[v][u] = c
                    adj.add(edge(u, v))
                # Try to add missing edges
                for (u, v) in all_pairs:
                    if not now_ok():
                        break
                    if edge(u, v) in adj:
                        continue
                    for c in range(_COLORS):
                        if evaluate_add(colmat, adj, u, v, c):
                            colmat[u][v] = c
                            colmat[v][u] = c
                            adj.add(edge(u, v))
                            break
                new_edges = []
                for (u, v) in sorted(adj):
                    new_edges.append([u, v, colmat[u][v]])
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best = new_edges

    # Also try natural order
    if now_ok():
        edges = try_build(list(all_pairs))
        if len(edges) > best_count:
            best_count = len(edges)
            best = edges

    if best is None:
        best = [[0, v, 0] for v in range(1, _N)]

    return best
