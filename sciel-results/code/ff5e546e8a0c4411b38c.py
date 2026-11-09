"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

N = 18
DEADLINE = 110.0  # safety margin under 120s


def _triangle_free_mask(adj):
    """Precompute for each vertex: bitmask of vertices that form no triangle with it.
    i.e., vertices v such that v is not adjacent to any neighbor of u (and v != u)."""
    tf = [0] * N
    for u in range(N):
        m = 0
        # vertices not adjacent to u
        for v in range(N):
            if v == u:
                continue
            if adj[u] & (1 << v):
                continue
            # check no common neighbor with u
            if (adj[u] & adj[v]) == 0:
                m |= (1 << v)
        tf[u] = m
    return tf


def _score(adj):
    """Exact branch-and-bound for max triangle-free induced subgraph size."""
    # Precompute: for a candidate set S (bitmask), a vertex v can be added iff
    # v has no neighbor in S (within S) -- i.e., v is non-adjacent to all in S.
    # Triangle-free means: no three mutually adjacent vertices.
    # We build S greedily/branch: S is triangle-free iff for every v in S,
    # S & adj[v] has no edges... simpler: S is triangle-free iff
    # for all v in S: (adj[v] & S) is an independent set.
    # Fast check: S triangle-free iff for all v in S, adj[v] & S & (union of adj of others) ... 
    # Use: S is triangle-free iff there is no triangle fully in S.
    # Precompute triangle bitmask: for each pair (u,v) with edge, common neighbors.
    # A set S contains a triangle iff exists edge (u,v) in S with common neighbor w in S.
    
    # Faster: precompute for each vertex v, the mask of edges among its neighbors.
    # S is triangle-free iff for every v in S, (adj[v] & S) is an independent set.
    
    # Independent set check for a set T: no edge within T.
    # Precompute edge masks: edge_mask[v] = bitmask of vertices adjacent to v.
    # T is independent iff for all v in T, adj[v] & T == 0... no, adj[v] & T should be 0
    # means v has no neighbor in T, but T includes v? No, T is the neighbor set.
    # T is independent iff for all v in T, (adj[v] & T) == 0.
    
    best = 0
    # Order vertices by degree for better pruning
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    
    def is_tf(S):
        """Check if set S (bitmask) is triangle-free."""
        # For each v in S, check adj[v] & S is independent
        s = S
        while s:
            lsb = s & (-s)
            v = lsb.bit_length() - 1
            s ^= lsb
            T = adj[v] & S
            # check T is independent
            t = T
            while t:
                lsb2 = t & (-t)
                w = lsb2.bit_length() - 1
                t ^= lsb2
                if adj[w] & T:
                    return False
        return True
    
    # Branch and bound
    # Upper bound: |S| + |candidates|
    # candidates: vertices that can be added to S while keeping triangle-free
    # A vertex v can be added to S iff adding v keeps S triangle-free,
    # i.e., S ∪ {v} is triangle-free.
    # This requires: (a) S is triangle-free (maintained), and
    # (b) v has no triangle with any pair in S, i.e., for all edges (a,b) in S, v not adjacent to both.
    # Equivalently: for all a in S, (adj[v] & adj[a] & S) == 0.
    
    # Precompute: for each v, and each a, the common neighbors mask.
    # can_add(v, S): for all a in S, (adj[v] & adj[a] & S) == 0
    
    def can_add(v, S):
        s = S
        while s:
            lsb = s & (-s)
            a = lsb.bit_length() - 1
            s ^= lsb
            if adj[v] & adj[a] & S:
                return False
        return True
    
    # Recursive search
    import sys
    sys.setrecursionlimit(10000)
    
    best = 0
    # Use iterative deepening or simple branch and bound
    # Start with empty set, try adding vertices in order
    
    def search(S, candidates, depth):
        nonlocal best
        if bin(S).count('1') > best:
            best = bin(S).count('1')
        if candidates == 0:
            return
        # Upper bound
        if bin(S).count('1') + bin(candidates).count('1') <= best:
            return
        # Try adding each candidate
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c ^= lsb
            # Check if v can be added
            if not can_add(v, S):
                continue
            new_S = S | (1 << v)
            # New candidates: those that can still be added
            new_c = 0
            cc = c  # remaining candidates (those after v in this iteration)
            while cc:
                lsb2 = cc & (-cc)
                w = lsb2.bit_length() - 1
                cc ^= lsb2
                if can_add(w, new_S):
                    new_c |= (1 << w)
            search(new_S, new_c, depth + 1)
    
    # Initial candidates: all vertices
    search(0, (1 << N) - 1, 0)
    return best


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _is_k4_free(adj):
    """Check if graph is K4-free using bitmask."""
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                # check if common has an edge
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[k] & common:
                        return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * N
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    _add_edge(adj, a, b)
    
    best_adj = adj[:]
    best_score = _score(adj)
    
    # Local search: try removing edges to reduce score
    # An edge (u,v) between parts i and j. Removing it might reduce triangle-free sets.
    # We want to find edges whose removal reduces the score.
    
    # Build list of all edges
    def get_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges
    
    # Local search loop
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 30:
        improved = False
        edges = get_edges(adj)
        rng.shuffle(edges)
        
        for (u, v) in edges:
            if (time.time() - start_time) > DEADLINE - 30:
                break
            # Try removing edge
            _remove_edge(adj, u, v)
            if _is_k4_free(adj):
                s = _score(adj)
                if s < best_score:
                    best_score = s
                    best_adj = adj[:]
                    improved = True
                else:
                    _add_edge(adj, u, v)
            else:
                _add_edge(adj, u, v)
    
    # Convert best_adj to edge list
    result = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
