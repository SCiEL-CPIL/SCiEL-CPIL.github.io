"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph.
   Uses bitmask-based scoring and simulated annealing with fast K4 checks."""

import random
import time
import math

N = 18
DEADLINE = 112.0  # safety margin before 120s limit


def _edge_set_to_edges(es):
    return [[u, v] for (u, v) in sorted(es)]


def _build_adj_mask(es):
    """Return adjacency as list of bitmasks."""
    adj = [0] * N
    for u, v in es:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _is_k4_free_mask(adj):
    """Check K4-free using adjacency masks. Fast: O(n^2 * degree)."""
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                # Check if any two vertices in common are adjacent
                # i.e., there exist w1 < w2 in common with w1 adjacent to w2
                mask = common
                while mask:
                    lb = mask & (-mask)
                    w1 = lb.bit_length() - 1
                    mask ^= lb
                    # Check if w1 is adjacent to any vertex in mask
                    if adj[w1] & mask:
                        return False
    return True


def _largest_triangle_free_mask(adj):
    """
    Find largest triangle-free induced subgraph using branch-and-bound with bitmasks.
    A set S is triangle-free iff for every edge (u,v) in S, no common neighbor of u,v is in S.
    Equivalent: no triangle fully contained in S.
    
    Strategy: recursive branch-and-bound. At each step, decide to include or exclude a vertex.
    Use upper bound: current_size + remaining_vertices.
    Prune if upper bound <= best found.
    """
    # Precompute: for each pair (i,j), the set of common neighbors (as bitmask)
    # A triangle (i,j,k) exists iff k is in common_neighbors[i][j]
    # For a candidate set S to be triangle-free: for every edge (i,j) in S, 
    #   common_neighbors[i][j] & S == 0 (excluding i,j themselves, but they're not in common anyway)
    
    # Actually simpler: a set S contains a triangle iff there exist i<j<k in S
    # such that all three edges exist.
    
    # Precompute triangle existence for each triple? Too many.
    
    # Better approach: maintain the set of "forbidden" vertices.
    # If we include vertex v, then for each pair (u,w) both adjacent to v and to each other,
    # we cannot include both u and w... this gets complex.
    
    # Let's use a different approach: enumerate by inclusion/exclusion with pruning.
    # Key insight: if S is not triangle-free, it contains a triangle.
    # We can use the fact that the complement of a triangle-free graph has no independent set of size 3.
    
    # Most efficient for n=18: use the property that a set is triangle-free iff
    # for each vertex v in S, the neighbors of v in S form an independent set.
    
    # Let's do branch and bound:
    best = [0]
    
    def bnb(candidates, current_size, current_mask):
        """candidates: bitmask of vertices still to consider, current_mask: vertices chosen so far"""
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound
        rem = candidates.bit_count()
        if current_size + rem <= best[0]:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with highest degree in remaining graph for better pruning
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Option 1: exclude v
        bnb(rest, current_size, current_mask)
        
        # Option 2: include v
        # Check: adding v to current_mask must not create a triangle
        # A triangle is formed if there exist u,w in current_mask such that 
        # u-v and v-w and u-w are all edges.
        # Equivalently: for each u in current_mask adjacent to v, 
        #   the neighbors of u in current_mask that are also adjacent to v form a conflict.
        # More precisely: let N_v = adj[v] & current_mask (neighbors of v already chosen)
        # For any two vertices u,w in N_v, if u-w is an edge, then {u,v,w} is a triangle.
        # So we need N_v to be an independent set in the induced subgraph.
        
        N_v = adj[v] & current_mask
        # Check if N_v is independent: no edge between any two vertices in N_v
        mask = N_v
        independent = True
        while mask:
            lb2 = mask & (-mask)
            u = lb2.bit_length() - 1
            mask ^= lb2
            if adj[u] & mask:
                independent = False
                break
        
        if independent:
            # Also need to ensure that adding v doesn't create triangles with future vertices
            # But that will be checked when those vertices are added.
            # However, we should restrict future candidates: any vertex w that forms a triangle
            # with v and some u in N_v should be excluded.
            # Specifically, for each u in N_v, w must not be adjacent to both v and u.
            # So forbidden = union over u in N_v of (adj[u] & adj[v])
            forbidden = 0
            m = N_v
            while m:
                lb3 = m & (-m)
                u = lb3.bit_length() - 1
                m ^= lb3
                forbidden |= (adj[u] & adj[v])
            # Remove forbidden vertices from rest
            new_rest = rest & ~forbidden
            bnb(new_rest, current_size + 1, current_mask | lb)
    
    bnb((1 << N) - 1, 0, 0)
    return best[0]


def _initial_graph():
    """Start with a random K4-free graph: complete tripartite."""
    parts = list(range(N))
    random.shuffle(parts)
    p0 = parts[:6]
    p1 = parts[6:12]
    p2 = parts[12:18]

    es = set()
    for a in p0:
        for b in p1:
            es.add((min(a,b), max(a,b)))
    for a in p0:
        for b in p2:
            es.add((min(a,b), max(a,b)))
    for a in p1:
        for b in p2:
            es.add((min(a,b), max(a,b)))
    return es


def _try_add_edge(es, adj):
    """Try to add a random edge that keeps graph K4-free. Returns new es or None."""
    all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
    non_edges = [e for e in all_possible if e not in es]
    if not non_edges:
        return None
    e = random.choice(non_edges)
    u, v = e
    # Adding (u,v) creates K4 iff there exist w1 < w2 in common neighbors of u,v
    # such that w1-w2 is an edge.
    common = adj[u] & adj[v]
    mask = common
    while mask:
        lb = mask & (-mask)
        w1 = lb.bit_length() - 1
        mask ^= lb
        if adj[w1] & mask:
            return None  # Would create K4
    new_es = set(es)
    new_es.add(e)
    return new_es


def _remove_random_edge(es):
    """Remove a random edge."""
    if not es:
        return None
    e = random.choice(list(es))
    new_es = set(es)
    new_es.discard(e)
    return new_es


def construct():
    start_time = time.time()
    random.seed(42)

    # Start with complete tripartite graph
    es = _initial_graph()
    adj = _build_adj_mask(es)
    best_score = _largest_triangle_free_mask(adj)
    best_es = set(es)

    current_temp = 2.0
    cooling = 0.999
    min_temp = 0.05

    while time.time() - start_time < DEADLINE:
        # Try a move
        new_es = None
        if random.random() < 0.4:
            new_es = _try_add_edge(es, adj)
        else:
            new_es = _remove_random_edge(es)
        
        if new_es is None or new_es == es:
            continue

        new_adj = _build_adj_mask(new_es)
        if not _is_k4_free_mask(new_adj):
            continue

        new_score = _largest_triangle_free_mask(new_adj)

        if new_score < best_score:
            best_score = new_score
            best_es = set(new_es)
            es = new_es
            adj = new_adj
        elif new_score <= best_score:
            es = new_es
            adj = new_adj
        else:
            delta = new_score - best_score
            if random.random() < math.exp(-delta / current_temp):
                es = new_es
                adj = new_adj

        current_temp = max(min_temp, current_temp * cooling)

        if best_score <= 5:
            break

    return _edge_set_to_edges(best_es)
