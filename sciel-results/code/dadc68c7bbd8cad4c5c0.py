"""Local search minimizing the largest book (max triangles per edge)."""
import time
import random
import math

N = 40
M = 401

def _bitcount(x):
    return x.bit_count()

def _make_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _book_of_edge(adj, u, v):
    # number of common neighbors = triangles containing edge (u,v)
    return _bitcount(adj[u] & adj[v])

def _max_book(adj):
    best = 0
    for u in range(N):
        au = adj[u]
        # iterate over set bits of au
        x = au
        while x:
            lsb = x & (-x)
            v = lsb.bit_length() - 1
            b = _bitcount(au & adj[v])
            if b > best:
                best = b
            x ^= lsb
    return best

def _max_book_and_edge(adj):
    best = 0
    be = None
    for u in range(N):
        au = adj[u]
        x = au
        while x:
            lsb = x & (-x)
            v = lsb.bit_length() - 1
            b = _bitcount(au & adj[v])
            if b > best:
                best = b
                be = (u, v)
            x ^= lsb
    return best, be

def _edge_key(e):
    u, v = e
    return (u, v) if u < v else (v, u)

def _initial_graph():
    # balanced bipartite 20+20 (400 edges) plus 1 extra edge chosen to
    # minimize the book it creates. Placing the extra edge inside part A
    # gives it book = |N(a)|&|N(b)| where both connect to all 20 in B -> 20.
    # Instead, we start from a near-bipartite graph and let search fix it.
    # Start: complete bipartite K20,20 plus one intra-part edge.
    edges = [[a, b] for a in range(20) for b in range(20, 40)]
    edges.append([0, 1])
    return edges

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s

    edges = _initial_graph()
    adj = _make_adj(edges)
    edge_set = set(_edge_key(e) for e in edges)
    cur_score = _max_book(adj)
    best_score = cur_score
    best_edges = list(edges)

    # Precompute all possible edges for proposal
    all_pairs = []
    for u in range(N):
        for v in range(u + 1, N):
            all_pairs.append((u, v))

    rng = random.Random(12345)
    temp = 5.0
    cooling = 0.9995

    while time.time() < deadline - 2.0:
        # propose: add an edge or remove an edge (keep exactly M edges)
        # choose add or remove based on current count (always M, so do swap)
        # Do an edge swap: remove one existing edge, add one non-existing edge.
        # Or single move: add edge (M+1) then remove edge (M). We do swap.

        # pick an existing edge to remove
        ri = rng.randrange(len(edges))
        ru, rv = edges[ri]
        # pick a non-existing edge to add
        for _ in range(50):
            au, av = all_pairs[rng.randrange(len(all_pairs))]
            if _edge_key((au, av)) not in edge_set:
                break
        else:
            continue

        # apply swap
        # remove (ru,rv)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        edge_set.discard(_edge_key((ru, rv)))
        # add (au,av)
        adj[au] |= 1 << av
        adj[av] |= 1 << au
        edge_set.add(_edge_key((au, av)))

        new_score = _max_book(adj)
        delta = new_score - cur_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            cur_score = new_score
            edges[ri] = [au, av]
            if cur_score < best_score:
                best_score = cur_score
                best_edges = list(edges)
        else:
            # revert
            adj[au] &= ~(1 << av)
            adj[av] &= ~(1 << au)
            edge_set.discard(_edge_key((au, av)))
            adj[ru] |= 1 << rv
            adj[rv] |= 1 << ru
            edge_set.add(_edge_key((ru, rv)))

        temp *= cooling
        if temp < 0.05:
            temp = 0.05

    # Return best found as list of [u,v]
    return [[u, v] for u, v in best_edges]
